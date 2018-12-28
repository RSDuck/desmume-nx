/*	Copyright (C) 2006 yopyop
	Copyright (C) 2011 Loren Merritt
	Copyright (C) 2012 DeSmuME team

	This file is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This file is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with the this software.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "types.h"

#ifdef HAVE_JIT

#include <unistd.h>
#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "instructions.h"
#include "instruction_attributes.h"
#include "MMU.h"
#include "MMU_timing.h"
#include "arm_jit.h"
#include "bios.h"
#include "armcpu.h"

#include "emitter/Arm64Emitter.h"
using namespace Arm64Gen;

namespace libnx {
#include <switch.h>
}
using namespace libnx;

/*
   ARM -> ARM64 JIT compiler

   Uses a fixed register mapping scheme:
      w12-w27: r0-r15
      w28: CPSR

      Per instruction scratch: w6-w9/x6-x10

      RCPU: x10
      RCYC: w11
*/

u32 saveBlockSizeJIT = 0;

#ifdef MAPPED_JIT_FUNCS
CACHE_ALIGN JIT_struct JIT;

uintptr_t *JIT_struct::JIT_MEM[2][0x4000] = {{0}};

static uintptr_t *JIT_MEM[2][32] = {
   //arm9
   {
      /* 0X*/  DUP2(JIT.ARM9_ITCM),
      /* 1X*/  DUP2(JIT.ARM9_ITCM), // mirror
      /* 2X*/  DUP2(JIT.MAIN_MEM),
      /* 3X*/  DUP2(JIT.SWIRAM),
      /* 4X*/  DUP2(NULL),
      /* 5X*/  DUP2(NULL),
      /* 6X*/      NULL,
                JIT.ARM9_LCDC,   // Plain ARM9-CPU Access (LCDC mode) (max 656KB)
      /* 7X*/  DUP2(NULL),
      /* 8X*/  DUP2(NULL),
      /* 9X*/  DUP2(NULL),
      /* AX*/  DUP2(NULL),
      /* BX*/  DUP2(NULL),
      /* CX*/  DUP2(NULL),
      /* DX*/  DUP2(NULL),
      /* EX*/  DUP2(NULL),
      /* FX*/  DUP2(JIT.ARM9_BIOS)
   },
   //arm7
   {
      /* 0X*/  DUP2(JIT.ARM7_BIOS),
      /* 1X*/  DUP2(NULL),
      /* 2X*/  DUP2(JIT.MAIN_MEM),
      /* 3X*/       JIT.SWIRAM,
                   JIT.ARM7_ERAM,
      /* 4X*/       NULL,
                   JIT.ARM7_WIRAM,
      /* 5X*/  DUP2(NULL),
      /* 6X*/      JIT.ARM7_WRAM,      // VRAM allocated as Work RAM to ARM7 (max. 256K)
                NULL,
      /* 7X*/  DUP2(NULL),
      /* 8X*/  DUP2(NULL),
      /* 9X*/  DUP2(NULL),
      /* AX*/  DUP2(NULL),
      /* BX*/  DUP2(NULL),
      /* CX*/  DUP2(NULL),
      /* DX*/  DUP2(NULL),
      /* EX*/  DUP2(NULL),
      /* FX*/  DUP2(NULL)
      }
};

static u32 JIT_MASK[2][32] = {
   //arm9
   {
      /* 0X*/  DUP2(0x00007FFF),
      /* 1X*/  DUP2(0x00007FFF),
      /* 2X*/  DUP2(0x003FFFFF), // FIXME _MMU_MAIN_MEM_MASK
      /* 3X*/  DUP2(0x00007FFF),
      /* 4X*/  DUP2(0x00000000),
      /* 5X*/  DUP2(0x00000000),
      /* 6X*/      0x00000000,
                0x000FFFFF,
      /* 7X*/  DUP2(0x00000000),
      /* 8X*/  DUP2(0x00000000),
      /* 9X*/  DUP2(0x00000000),
      /* AX*/  DUP2(0x00000000),
      /* BX*/  DUP2(0x00000000),
      /* CX*/  DUP2(0x00000000),
      /* DX*/  DUP2(0x00000000),
      /* EX*/  DUP2(0x00000000),
      /* FX*/  DUP2(0x00007FFF)
   },
   //arm7
   {
      /* 0X*/  DUP2(0x00003FFF),
      /* 1X*/  DUP2(0x00000000),
      /* 2X*/  DUP2(0x003FFFFF),
      /* 3X*/       0x00007FFF,
                   0x0000FFFF,
      /* 4X*/       0x00000000,
                   0x0000FFFF,
      /* 5X*/  DUP2(0x00000000),
      /* 6X*/      0x0003FFFF,
                0x00000000,
      /* 7X*/  DUP2(0x00000000),
      /* 8X*/  DUP2(0x00000000),
      /* 9X*/  DUP2(0x00000000),
      /* AX*/  DUP2(0x00000000),
      /* BX*/  DUP2(0x00000000),
      /* CX*/  DUP2(0x00000000),
      /* DX*/  DUP2(0x00000000),
      /* EX*/  DUP2(0x00000000),
      /* FX*/  DUP2(0x00000000)
      }
};

static void init_jit_mem()
{
   static bool inited = false;
   if(inited)
      return;
   inited = true;
   for(int proc=0; proc<2; proc++)
      for(int i=0; i<0x4000; i++)
         JIT.JIT_MEM[proc][i] = JIT_MEM[proc][i>>9] + (((i<<14) & JIT_MASK[proc][i>>9]) >> 1);
}

#else
DS_ALIGN(4096) uintptr_t compiled_funcs[1<<26] = {0};
#endif

template<int PROCNUM, int thumb>
static u32 FASTCALL OP_DECODE()
{
   u32 cycles;
   u32 adr = ARMPROC.instruct_adr;
   if(thumb)
   {
      ARMPROC.next_instruction = adr + 2;
      ARMPROC.R[15] = adr + 4;
      u32 opcode = _MMU_read16<PROCNUM, MMU_AT_CODE>(adr);
      //_armlog(PROCNUM, adr, opcode);
      cycles = thumb_instructions_set[PROCNUM][opcode>>6](opcode);
   }
   else
   {
      ARMPROC.next_instruction = adr + 4;
      ARMPROC.R[15] = adr + 8;
      u32 opcode = _MMU_read32<PROCNUM, MMU_AT_CODE>(adr);
      //_armlog(PROCNUM, adr, opcode);
      if(CONDITION(opcode) == 0xE || TEST_COND(CONDITION(opcode), CODE(opcode), ARMPROC.CPSR))
         cycles = arm_instructions_set[PROCNUM][INSTRUCTION_INDEX(opcode)](opcode);
      else
         cycles = 1;
   }
   ARMPROC.instruct_adr = ARMPROC.next_instruction;
   return cycles;
}

static const ArmOpCompiled op_decode[2][2] = { OP_DECODE<0,0>, OP_DECODE<0,1>, OP_DECODE<1,0>, OP_DECODE<1,1> };


enum OP_RESULT { OPR_CONTINUE, OPR_INTERPRET, OPR_BRANCHED, OPR_RESULT_SIZE = 2147483647 };
#define OPR_RESULT(result, cycles) (OP_RESULT)((result) | ((cycles) << 16));
#define OPR_RESULT_CYCLES(result) ((result >> 16))
#define OPR_RESULT_ACTION(result) ((result & 0xFF))

typedef OP_RESULT (*ArmOpCompiler)(uint32_t pc, uint32_t opcode);

static const uint32_t INSTRUCTION_COUNT = 0xC0000;

static libnx::Jit jitPage;
static u8* jitRWAddr = nullptr;
static u8* jitRXAddr = nullptr;
static int jitUsed = -1;
static const int JitPageSize = 0x100000;

static ARM64XEmitter emit;

static u8 recompile_counts[(1<<26)/16];

const ARM64Reg RCPU = X10;
const ARM64Reg RCYC = W11;

enum
{
   FNegativeShift = 31,
   FNegativeBit = 1 << FNegativeShift,
   FZeroShift = 30,
   FZeroBit = 1 << FZeroShift,
   FCarryShift = 29,
   FCarryBit = 1 << FCarryShift,
   FOverflowShift = 28,
   FOverflowBit = 1 << FOverflowShift,
   FQShift = 27,
   FQBit = 1 << FQShift,
};

const ARM64Reg RCPSR = W28;
static uint32_t block_procnum;

///////
// HELPERS
///////
class reg_manager
{
   int emu_regs[8]; // [native register] = emulated register
   int reg_usage[8];
   u8 regs_dirty;

   int native_regs[16]; // [emulated register] = native register

   ARM64Reg get_reg_n(int n)
   {
      return (ARM64Reg)(W19 + n);
   }

   int get_least_used()
   {
      int usage = INT_MAX;
      int reg = -1;
      for (int i = 0; i < 8; i++)
      {
         if (reg_usage[i] < usage) {
            usage = reg_usage[i];
            reg = i;
         }
      }
      return reg;
   }
public:
   reg_manager()
   {
      reset();
   }

   void reset()
   {
      for (int i = 0; i < 8; i++)
         emu_regs[i] = reg_usage[i] = -1;
      for (int i = 0; i < 16; i++)
         native_regs[i] = -1;
      
      regs_dirty = 0;
   }

   void flush()
   {
      for (int i = 0; i < 8; i++)
         if (regs_dirty & (1 << i))
            emit.STR(INDEX_UNSIGNED, get_reg_n(i), RCPU, offsetof(armcpu_t, R) + 4 * emu_regs[i]);
      regs_dirty = 0;
   }

   ARM64Reg alloc(int reg, bool mut = false)
   {
      int native_reg = -1;
      if (native_regs[reg] != -1)
         native_reg = native_regs[reg];
      else
      {
         native_reg = get_least_used();

         if (emu_regs[native_reg] != -1) {
            if (regs_dirty & (1 << native_reg)) {
               emit.STR(INDEX_UNSIGNED, get_reg_n(native_reg), RCPU, offsetof(armcpu_t, R) + 4 * emu_regs[native_reg]);
               regs_dirty &= ~(1 << native_reg);
            }
            native_regs[emu_regs[native_reg]] = -1;
         }

         emit.LDR(INDEX_UNSIGNED, get_reg_n(native_reg), RCPU, offsetof(armcpu_t, R) + 4 * reg);

         native_regs[reg] = native_reg;
         emu_regs[native_reg] = reg;
         reg_usage[native_reg] = 0;
      }

      regs_dirty |= (mut << native_reg);

      reg_usage[native_reg]++;

      return get_reg_n(native_reg);
   }
};
static reg_manager regman;

static bool hw_status_dirty;
static bool emu_status_dirty;

static bool bit(uint32_t value, uint32_t bit)
{
   return value & (1 << bit);
}

static uint32_t bit(uint32_t value, uint32_t first, uint32_t count)
{
   return (value >> first) & ((1 << count) - 1);
}

static uint32_t bit_write(uint32_t value, uint32_t first, uint32_t count, uint32_t insert)
{
   uint32_t result = value & ~(((1 << count) - 1) << first);
   return result | (insert << first);
}

static ARM64Reg to_64bit_reg(ARM64Reg reg) {
   return static_cast<ARM64Reg>(reg - W0 + X0);
}
static ARM64Reg to_32bit_reg(ARM64Reg reg) {
   return static_cast<ARM64Reg>(reg - X0 + W0);
}

static void load_status_from_mem()
{
   emit.LDR(INDEX_UNSIGNED, RCPSR, RCPU, offsetof(armcpu_t, CPSR));
   hw_status_dirty = true;
}
static void write_status_to_mem()
{
   emit.STR(INDEX_UNSIGNED, RCPSR, RCPU, offsetof(armcpu_t, CPSR));
}
static void flush_hwstatus()
{
   if (hw_status_dirty)
   {
      emit._MSR(FIELD_NZCV, to_64bit_reg(RCPSR));
      hw_status_dirty = false;
   }
}

static void copy_overflowq()
{
   emit.CSET(W6, CC_VS);
   emit.ORR(RCPSR, RCPSR, W6, ArithOption(W6, ST_LSL, FQShift));
   hw_status_dirty = true;
}

static void copy_hwstatus(bool overflow, bool carry, bool zeroflag, bool negative)
{
   if (overflow && carry && zeroflag && negative)
   {
      emit.MRS(X6, FIELD_NZCV);
      emit.UBFX(W6, W6, 28, 4);
      emit.BFI(RCPSR, W6, 28, 4);
   }
   else
   {
      if (overflow)
      {
         emit.CSET(W6, CC_VS);
         emit.BFI(RCPSR, W6, FOverflowShift, 1);
      }
      if (carry)
      {
         emit.CSET(W6, CC_CS);
         emit.BFI(RCPSR, W6, FCarryShift, 1);
      }
      if (zeroflag)
      {
         emit.CSET(W6, CC_EQ);
         emit.BFI(RCPSR, W6, FZeroShift, 1);
      }
      if (negative)
      {
         emit.CSET(W6, CC_MI);
         emit.BFI(RCPSR, W6, FNegativeShift, 1);
      }
      hw_status_dirty = true;
   }
}

static void close_branch(std::optional<FixupBranch> branch)
{
   if (branch.has_value())
   {
      emit.SetJumpTarget(*branch);
   }
}

static std::optional<FixupBranch> branch_impl(int cond)
{
   if(cond != CC_AL) {
      auto skip = emit.B((CCFlags)cond);
      auto res = emit.B(CC_AL);
      emit.SetJumpTarget(skip);
      return res;
   }
   return std::optional<FixupBranch>();
}

static void change_mode(bool thumb)
{
   if (!thumb)
   {
      emit.AND(RCPSR, RCPSR, 0x0, 0x10, true);
   }
   else
   {
      emit.ORR(RCPSR, RCPSR, 0x0, 0x10);
   }

   hw_status_dirty = true;

   write_status_to_mem();
}

template <typename T>
static void call(ARM64Reg scratch, T func)
{
   emit.ABI_PushRegisters({10, 11}); // RCPU, RCYC
   emit.QuickCallFunction(scratch, func);
   emit.ABI_PopRegisters({10, 11});

   hw_status_dirty = true;
}
/*
static void change_mode_reg(reg_t reg, reg_t scratch, reg_t scratch2)
{
   block->and_(scratch2, reg, alu2::imm(1));

   block->ldr(scratch, RCPU, mem2::imm(offsetof(armcpu_t, CPSR)));
   block->bic(scratch, alu2::imm(scratch2 << 5));
   block->orr(scratch, alu2::reg_shift_imm(scratch2, LSL, 5));
   block->str(scratch, RCPU, mem2::imm(offsetof(armcpu_t, CPSR)));
}*/

template <int PROCNUM>
static void arm_jit_prefetch(uint32_t pc, uint32_t opcode, bool thumb)
{
   const uint32_t imask = thumb ? 0xFFFFFFFE : 0xFFFFFFFC;
   const uint32_t isize = thumb ? 2 : 4;

   emit.MOVI2R(W0, pc & imask);
   emit.STR(INDEX_UNSIGNED, W0, RCPU, offsetof(armcpu_t, instruct_adr));
   
   emit.ADD(W0, W0, isize);
   emit.STR(INDEX_UNSIGNED, W0, RCPU, offsetof(armcpu_t, next_instruction));

   emit.ADD(W0, W0, isize);
   emit.STR(INDEX_UNSIGNED, W0, RCPU, offsetof(armcpu_t, R) + 4 * 15);

   emit.MOVI2R(W0, opcode);
   emit.STR(INDEX_UNSIGNED, W0, RCPU, offsetof(armcpu_t, instruction));
}

/////////
/// ARM
/////////

/*
      Instruktionen wo rhs Shift um Immediate ist
            => direkt übersetzen

*/
// welcome to pre processor hell…

#define ARM_ALU_OP_DEF(name, body, cyc, has_rd, has_rn, has_rm, has_rs) \
   static OP_RESULT ARM_OP_##name(uint32_t opcode, uint32_t pc) { \
      has_rd( \
         const int rd = bit(opcode, 12, 4); \
         if (rd == 0xf) return OPR_INTERPRET; \
         const ARM64Reg rd_native = regman.alloc(rd, true); \
      ) \
      has_rn( \
         const int rn = bit(opcode, 16, 4); \
         if (rn == 0xf) return OPR_INTERPRET; /*emit.MOVI2R(rn_native, pc + 8);*/ \
         const ARM64Reg rn_native = regman.alloc(rn); \
      ) \
      has_rm( \
         const int rm = bit(opcode, 0, 4); \
         if (rm == 0xf) return OPR_INTERPRET; /*emit.MOVI2R(rm_native, pc + 8);*/ \
         const ARM64Reg rm_native = regman.alloc(rm); \
      ) \
      has_rs( \
         const int rs = bit(opcode, 8, 4); \
         const ARM64Reg rs_native = regman.alloc(rs); \
      ) \
      const int cond = bit(opcode, 28, 4); \
      auto condBranch = branch_impl(cond); \
      body \
      close_branch(condBranch); \
      return OPR_RESULT(OPR_CONTINUE, cyc); \
   }

#define DEF_ARM_ALU_OP_IMM_SHIFT(name, op) \
      ARM_ALU_OP_DEF(name, 



const ARM64Reg RALUSrc2 = W9;

#define SHIFT_IMM_INTRO \
   const int shiftAmount = bit(opcode, 7, 5);
#define SHIFT_REG_INTRO 

#define LSL_IMM \
   SHIFT_IMM_INTRO \
   emit.MOV(RALUSrc2, rm_native, ArithOption(W9, ST_LSL, shiftAmount));
#define LSL_IMM_S \
   return OPR_INTERPRET; \
   LSL_IMM \
   if(shiftAmount != 0) { \
      emit.UBFX(W6, rm_native, 32 - shiftAmount, 1); \
      emit.BFI(RCPSR, W6, FCarryShift, 1); \
      hw_status_dirty = true; \
   }

#define LSL_REG \
   SHIFT_REG_INTRO \
   emit.LSLV(RALUSrc2, rm_native, rs_native);
#define LSL_REG_S \
   return OPR_INTERPRET; \
   LSL_REG \
   FixupBranch branch = emit.CBZ(rs_native); \
   emit.MOVZ(W6, 32); \
   emit.SUB(W6, W6, rs_native); \
   emit.LSRV(W6, rm_native, W6); \
   emit.BFI(RCPSR, W6, FCarryShift, 1); \
   emit.SetJumpTarget(branch); \
   hw_status_dirty = true;

#define LSR_IMM \
   SHIFT_IMM_INTRO \
   if(shiftAmount != 0) \
      emit.MOV(RALUSrc2, rm_native, ArithOption(RALUSrc2, ST_LSR, shiftAmount)); \
   else \
      emit.MOVZ(RALUSrc2, 0);
#define LSR_IMM_S \
   return OPR_INTERPRET; \
   LSR_IMM \
   if(shiftAmount != 0) \
      emit.UBFX(W6, rm_native, shiftAmount - 1, 1); \
   else \
      emit.MOVZ(W6, 0); \
   emit.BFI(RCPSR, W6, FCarryShift, 1); \
   hw_status_dirty = true;

#define LSR_REG \
   SHIFT_REG_INTRO \
   emit.LSRV(RALUSrc2, rm_native, rs_native);
#define LSR_REG_S \
   return OPR_INTERPRET; \
   LSR_REG \
   FixupBranch branch = emit.CBZ(rs_native); \
   emit.SUB(W6, rs_native, 1); \
   emit.LSRV(W7, rm_native, W6); \
   emit.BFI(RCPSR, W7, FCarryShift, 1); \
   emit.SetJumpTarget(branch); \
   hw_status_dirty = true;

#define ASR_IMM \
   SHIFT_IMM_INTRO \
   if(shiftAmount != 0) \
      emit.MOV(RALUSrc2, rm_native, ArithOption(RALUSrc2, ST_ASR, shiftAmount)); \
   else { \
      emit.UBFX(W7, rm_native, 31, 1); \
      emit.CMP(W7, 0); \
      emit.MOVZ(W6, 0); \
      emit.CSINV(RALUSrc2, W6, W6, CC_EQ); \
      hw_status_dirty = true; \
   }
#define ASR_IMM_S \
   return OPR_INTERPRET; \
   ASR_IMM \
   if(shiftAmount != 0) \
      emit.UBFX(W7, rm_native, shiftAmount - 1, 1); \
   emit.BFI(RCPSR, W7, FCarryShift, 1); // assuming W7 is still set to rm[31] if shiftAmount != 0

#define ASR_REG \
   SHIFT_REG_INTRO \
   emit.ASRV(RALUSrc2, rm_native, rs_native);
#define ASR_REG_S \
   return OPR_INTERPRET; \
   ASR_REG \
   FixupBranch branch = emit.CBZ(rs_native); \
   emit.SUB(W6, rs_native, 1); \
   emit.LSRV(W7, rm_native, W6); \
   emit.BFI(RCPSR, W7, FCarryShift, 1); \
   emit.SetJumpTarget(branch); \
   hw_status_dirty = true;

#define ROR_IMM \
   SHIFT_IMM_INTRO \
   if(shiftAmount != 0) \
      emit.MOV(RALUSrc2, rm_native, ArithOption(RALUSrc2, ST_ROR, shiftAmount)); \
   else { \
      emit.MOV(RALUSrc2, rm_native, ArithOption(RALUSrc2, ST_ASR, 1)); \
      emit.UBFX(W6, RCPSR, FCarryShift, 1); \
      emit.BFI(RALUSrc2, W6, 31, 1); \
   }
#define ROR_IMM_S \
   return OPR_INTERPRET; \
   ROR_IMM \
   if(shiftAmount != 0) { \
      emit.UBFX(W7, rm_native, shiftAmount - 1, 1); \
      emit.BFI(RCPSR, W7, FCarryShift, 1); \
   } else \
      emit.BFI(RCPSR, rm_native, FCarryShift, 1); \
   hw_status_dirty = true;

#define ROR_REG \
   SHIFT_REG_INTRO \
   emit.RORV(RALUSrc2, rm_native, rs_native);
#define ROR_REG_S \
   return OPR_INTERPRET; \
   ROR_REG \
   FixupBranch branch = emit.CBZ(rs_native); \
   emit.SUB(W6, rs_native, 1); \
   emit.LSRV(W7, rm_native, W6); \
   emit.BFI(RCPSR, W7, FCarryShift, 1); \
   emit.SetJumpTarget(branch); \
   hw_status_dirty = true;

#define IMM_VAL \
   const int immVal = bit(opcode, 0, 8); \
   const int shiftAmount = bit(opcode, 8, 4); \
   emit.MOVZ(RALUSrc2, immVal); \
   if(shiftAmount > 0) \
      emit.ROR(RALUSrc2, RALUSrc2, shiftAmount);

#define use(x) x
#define discard(x)
#define ARM_ALU_OP_DEFS(T, op, has_rd, has_rn, has_rm) \
   ARM_ALU_OP_DEF(T##_LSL_IMM, LSL_IMM op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_LSL_REG, LSL_REG op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_LSR_IMM, LSR_IMM op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_LSR_REG, LSR_REG op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_ASR_IMM, ASR_IMM op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_ASR_REG, ASR_REG op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_ROR_IMM, ROR_IMM op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_ROR_REG, ROR_REG op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_IMM_VAL, IMM_VAL op, 1, has_rd, has_rn, discard, discard)
#define ARM_ALU_OP_DEFS_S(T, op, has_rd, has_rn, has_rm) \
   ARM_ALU_OP_DEF(T##_LSL_IMM, LSL_IMM_S op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_LSL_REG, LSL_REG_S op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_LSR_IMM, LSR_IMM_S op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_LSR_REG, LSR_REG_S op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_ASR_IMM, ASR_IMM_S op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_ASR_REG, ASR_REG_S op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF(T##_ROR_IMM, ROR_IMM_S op, 1, has_rd, has_rn, has_rm, discard) \
   ARM_ALU_OP_DEF(T##_ROR_REG, ROR_REG_S op, 2, has_rd, has_rn, has_rm, use) \
   ARM_ALU_OP_DEF (T##_IMM_VAL, IMM_VAL op, 1, has_rd, has_rn, discard, discard)

ARM_ALU_OP_DEFS(AND, emit.AND(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS_S(AND_S, emit.ANDS(rd_native, rn_native, RALUSrc2); copy_hwstatus(false, false, true, true);, use, use, use)
ARM_ALU_OP_DEFS(EOR, emit.EOR(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS_S(EOR_S, emit.EOR(rd_native, rn_native, RALUSrc2); emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, use, use, use)
ARM_ALU_OP_DEFS(SUB, emit.SUB(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS(SUB_S, return OPR_INTERPRET; emit.SUBS(rd_native, rn_native, RALUSrc2); copy_hwstatus(true, true, true, true);, use, use, use)
ARM_ALU_OP_DEFS(RSB, emit.SUB(rd_native, RALUSrc2, rn_native);, use, use, use)
ARM_ALU_OP_DEFS(RSB_S, return OPR_INTERPRET; emit.SUBS(rd_native, RALUSrc2, rn_native); copy_hwstatus(true, true, true, true);, use, use, use)
ARM_ALU_OP_DEFS(ADD, emit.ADD(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS(ADD_S, return OPR_INTERPRET; emit.ADDS(rd_native, rn_native, RALUSrc2); copy_hwstatus(true, true, true, true);, use, use, use)
ARM_ALU_OP_DEFS(ADC, return OPR_INTERPRET; flush_hwstatus(); emit.ADC(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS(ADC_S, return OPR_INTERPRET; flush_hwstatus(); emit.ADCS(rd_native, rn_native, RALUSrc2); copy_hwstatus(true, true, true, true);, use, use, use)
ARM_ALU_OP_DEFS(SBC, return OPR_INTERPRET; flush_hwstatus(); emit.SBC(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS(SBC_S, return OPR_INTERPRET; flush_hwstatus(); emit.SBCS(rd_native, rn_native, RALUSrc2); copy_hwstatus(true, true, true, true);, use, use, use)
ARM_ALU_OP_DEFS(RSC, return OPR_INTERPRET; flush_hwstatus(); emit.SBC(rd_native, RALUSrc2, rn_native);, use, use, use)
ARM_ALU_OP_DEFS(RSC_S, return OPR_INTERPRET; flush_hwstatus(); emit.SBCS(rd_native, RALUSrc2, rn_native); copy_hwstatus(true, true, true, true);, use, use, use)
ARM_ALU_OP_DEFS_S(TST, emit.TST(rn_native, RALUSrc2); copy_hwstatus(false, false, true, true);, discard, use, use)
ARM_ALU_OP_DEFS_S(TEQ, emit.EOR(W6, rn_native, RALUSrc2); emit.TST(W6, W6); copy_hwstatus(false, false, true, true);, discard, use, use)
ARM_ALU_OP_DEFS(CMP, emit.CMP(rn_native, RALUSrc2); copy_hwstatus(true, true, true, true);, discard, use, use)
ARM_ALU_OP_DEFS(CMN, emit.CMN(rn_native, RALUSrc2); copy_hwstatus(true, true, true, true);, discard, use, use)
ARM_ALU_OP_DEFS(ORR, emit.ORR(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS_S(ORR_S, emit.ORR(rd_native, rn_native, RALUSrc2); emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, use, use, use)
ARM_ALU_OP_DEFS(MOV, emit.MOV(rd_native, RALUSrc2);, use, discard, use)
ARM_ALU_OP_DEFS_S(MOV_S, emit.MOV(rd_native, RALUSrc2); emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, use, discard, use)
ARM_ALU_OP_DEFS(BIC, emit.BIC(rd_native, rn_native, RALUSrc2);, use, use, use)
ARM_ALU_OP_DEFS_S(BIC_S, emit.BICS(rd_native, rn_native, RALUSrc2); copy_hwstatus(false, false, true, true);, use, use, use);
ARM_ALU_OP_DEFS(MVN, emit.MVN(rd_native, RALUSrc2);, use, discard, use)
ARM_ALU_OP_DEFS_S(MVN_S, emit.MVN(rd_native, RALUSrc2); emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, use, discard, use);

#define ARM_MUL_OP_DEF(name, op, cyc, use_rn, rn_mut) \
   static OP_RESULT ARM_OP_##name(uint32_t opcode, uint32_t pc) { \
      return OPR_INTERPRET; \
      const ARM64Reg rd = regman.alloc(bit(opcode, 16, 4), true); \
      use_rn(const ARM64Reg rn = regman.alloc(bit(opcode, 12, 4), rn_mut);) \
      const ARM64Reg rs = regman.alloc(bit(opcode, 8, 4)); \
      const ARM64Reg rm = regman.alloc(bit(opcode, 0, 4)); \
      const int cond = bit(opcode, 28, 4); \
      auto condBranch = branch_impl(cond); \
      op \
      close_branch(condBranch); \
      return OPR_RESULT(OPR_CONTINUE, cyc); \
   }

#define MUL_SPLIT_X6 \
   emit.MOV(rn, W6); \
   emit.MOV(rd, X6, ArithOption(rd, ST_LSR, 32));

// HACK: multiply cycles are (still in ARM64) wrong
// MUL
ARM_MUL_OP_DEF(MUL, emit.MUL(to_64bit_reg(rd), rm, rs); emit.MOV(rd, rd);, 3, discard, false)
ARM_MUL_OP_DEF(MUL_S, 
   const ARM64Reg rd64 = to_64bit_reg(rd); 
   emit.MUL(rd64, rm, rs); 
   emit.MOV(rd, rd);
   emit.TST(rd64, rd64); 
   copy_hwstatus(false, false, true, true);, 3, discard, false)
ARM_MUL_OP_DEF(MLA, emit.MADD(to_64bit_reg(rd), rm, rs, rn); emit.MOV(rd, rd);, 4, use, false)
ARM_MUL_OP_DEF(MLA_S, 
   const ARM64Reg rd64 = to_64bit_reg(rd); 
   emit.MADD(rd64, rm, rs, rn);
   emit.MOV(rd, rd);
   emit.TST(rd64, rd64); 
   copy_hwstatus(false, false, true, true);, 4, use, false)
// UMULL
ARM_MUL_OP_DEF(UMULL, emit.UMULL(X6, rm, rs); MUL_SPLIT_X6, 4, use, true)
ARM_MUL_OP_DEF(UMULL_S,
   emit.UMULL(X6, rm, rs); 
   MUL_SPLIT_X6
   emit.TST(X6, X6);
   copy_hwstatus(false, false, true, true);, 4, use, true)

#define MUL_COMBINE_rdHILO(reg) \
   emit.MOV(reg, rn); \
   emit.MOV(reg, rd, ArithOption(reg, ST_LSL, 32));

ARM_MUL_OP_DEF(UMLAL,
   MUL_COMBINE_rdHILO(W7)
   emit.UMADDL(X6, rm, rs, X7);
   MUL_SPLIT_X6, 5, use, true)
ARM_MUL_OP_DEF(UMLAL_S,
   MUL_COMBINE_rdHILO(W7)
   emit.UMADDL(X6, rm, rs, X7);
   emit.TST(X6, X6);
   copy_hwstatus(false, false, true, true);
   MUL_SPLIT_X6, 5, use, true)
// SMULL
ARM_MUL_OP_DEF(SMULL, 
   emit.SMULL(X6, rm, rs); 
   MUL_SPLIT_X6, 4, use, true)
ARM_MUL_OP_DEF(SMULL_S,
   emit.SMULH(X6, rm, rs); 
   MUL_SPLIT_X6
   emit.TST(X6, X6);
   copy_hwstatus(false, false, true, true);, 4, use, true)
ARM_MUL_OP_DEF(SMLAL,
   MUL_COMBINE_rdHILO(W7)
   emit.SMADDL(X6, rm, rs, X7);
   MUL_SPLIT_X6, 5, use, true)
ARM_MUL_OP_DEF(SMLAL_S,
   MUL_COMBINE_rdHILO(W7)
   emit.SMADDL(X6, rm, rs, X7);
   emit.TST(X6, X6);
   copy_hwstatus(false, false, true, true);
   MUL_SPLIT_X6, 5, use, true)

#define MUL_LOAD_B_RM \
   emit.SXTH(W6, rm);
#define MUL_LOAD_T_RM \
   emit.SBFM(W6, rm, 16, 16);
#define MUL_LOAD_B_RS \
   emit.SXTH(W7, rs);
#define MUL_LOAD_T_RS \
   emit.SBFM(W7, rs, 16, 16);

#define MUL_OP_DEF_HALF(name, op, use_rn, rn_mut) \
   ARM_MUL_OP_DEF(name##_B_B, MUL_LOAD_B_RM MUL_LOAD_B_RS op, 4, use_rn, rn_mut) \
   ARM_MUL_OP_DEF(name##_T_B, MUL_LOAD_T_RM MUL_LOAD_B_RS op, 4, use_rn, rn_mut) \
   ARM_MUL_OP_DEF(name##_B_T, MUL_LOAD_B_RM MUL_LOAD_T_RS op, 4, use_rn, rn_mut) \
   ARM_MUL_OP_DEF(name##_T_T, MUL_LOAD_T_RM MUL_LOAD_T_RS op, 4, use_rn, rn_mut)

MUL_OP_DEF_HALF(SMUL, emit.MUL(rd, W6, W7);, discard, false)
MUL_OP_DEF_HALF(SMLA, emit.MADD(rd, W6, W7, rn);, use, false)

ARM_MUL_OP_DEF(SMULW_B, MUL_LOAD_B_RS emit.MUL(rd, rm, W6); emit.ASR(rd, rd, 4);, 2, discard, false)
ARM_MUL_OP_DEF(SMULW_T, MUL_LOAD_T_RS emit.MUL(rd, rm, W6); emit.ASR(rd, rd, 4);, 2, discard, false)
ARM_MUL_OP_DEF(SMLAW_B, 
   MUL_LOAD_B_RS 
   emit.MUL(rd, rm, W6); 
   emit.ADDS(rd, rd, rn, ArithOption(rd, ST_ASR, 4));
   copy_overflowq();
   hw_status_dirty = true;, 2, use, false)
ARM_MUL_OP_DEF(SMLAW_T, 
   MUL_LOAD_T_RS 
   emit.MUL(rd, rm, W6); 
   emit.ADDS(rd, rd, rn, ArithOption(rd, ST_ASR, 4));
   copy_overflowq();
   hw_status_dirty = true;, 2, use, false)

MUL_OP_DEF_HALF(SMLAL, MUL_COMBINE_rdHILO(W8) emit.SMADDL(X6, W6, W7, W8); MUL_SPLIT_X6, use, true)

#define ARM_OP_Q_DEF(name, op, use_rn) \
   static OP_RESULT ARM_OP_##name(uint32_t opcode, uint32_t pc) { \
      return OPR_INTERPRET; \
      const ARM64Reg rd = regman.alloc(bit(opcode, 15, 4), true); \
      use_rn(const ARM64Reg rn = regman.alloc(bit(opcode, 19, 4))); \
      const ARM64Reg rm = regman.alloc(bit(opcode, 0, 4)); \
      const int cond = bit(opcode, 28, 4); \
      auto condBranch = branch_impl(cond); \
      op \
      close_branch(condBranch); \
      return OPR_RESULT(OPR_CONTINUE, 1); \
   }

#define ARM_OP_Q_SETUP_TRUNC_VALS(reg) \
   emit.UBFX(W7, rd, 31, 1); \
   emit.MOVI2R(W6, 0x80000000); \
   emit.SUB(reg, W6, W7);
#define ARM_OP_Q_TRUNC(reg) \
   emit.CSEL(rd, reg, rd, CC_VS);

ARM_OP_Q_DEF(QADD, 
   emit.ADDS(rd, rn, rm); 
   ARM_OP_Q_SETUP_TRUNC_VALS(W6)
   ARM_OP_Q_TRUNC(W6)
   copy_overflowq();
   hw_status_dirty = true;, use)
ARM_OP_Q_DEF(QSUB, 
   emit.SUBS(rd, rn, rm); 
   ARM_OP_Q_SETUP_TRUNC_VALS(W6)
   ARM_OP_Q_TRUNC(W6)
   copy_overflowq();
   hw_status_dirty = true;, use)
ARM_OP_Q_DEF(QDADD,
   emit.MOV(rd, rn);
   emit.ADDS(rd, rd, rn);
   ARM_OP_Q_SETUP_TRUNC_VALS(W6)
   ARM_OP_Q_TRUNC(W6)
   copy_overflowq();
   emit.ADDS(rd, rd, rm);
   ARM_OP_Q_TRUNC(W6);
   copy_overflowq();
   hw_status_dirty = true;, use)
ARM_OP_Q_DEF(QDSUB,
   emit.MOV(rd, rn);
   emit.ADDS(rd, rd, rn);
   ARM_OP_Q_SETUP_TRUNC_VALS(W6)
   ARM_OP_Q_TRUNC(W6)
   copy_overflowq();
   emit.SUBS(rd, rd, rm);
   ARM_OP_Q_TRUNC(W6);
   copy_overflowq();
   hw_status_dirty = true;, use)
ARM_OP_Q_DEF(CLZ, emit.CLZ(rd, rm);, discard)

////////
// Need versions of these functions with exported symbol
u8  _MMU_read08_9(u32 addr) { return _MMU_read08<0>(addr); }
u8  _MMU_read08_7(u32 addr) { return _MMU_read08<1>(addr); }
u16 _MMU_read16_9(u32 addr) { return _MMU_read16<0>(addr & 0xFFFFFFFE); }
u16 _MMU_read16_7(u32 addr) { return _MMU_read16<1>(addr & 0xFFFFFFFE); }
u32 _MMU_read32_9(u32 addr) { return ::ROR(_MMU_read32<0>(addr & 0xFFFFFFFC), 8 * (addr & 3)); }
u32 _MMU_read32_7(u32 addr) { return ::ROR(_MMU_read32<1>(addr & 0xFFFFFFFC), 8 * (addr & 3)); }

void _MMU_write08_9(u32 addr, u8  val) { _MMU_write08<0>(addr, val); }
void _MMU_write08_7(u32 addr, u8  val) { _MMU_write08<1>(addr, val); }
void _MMU_write16_9(u32 addr, u16 val) { _MMU_write16<0>(addr & 0xFFFFFFFE, val); }
void _MMU_write16_7(u32 addr, u16 val) { _MMU_write16<1>(addr & 0xFFFFFFFE, val); }
void _MMU_write32_9(u32 addr, u32 val) { _MMU_write32<0>(addr & 0xFFFFFFFC, val); }
void _MMU_write32_7(u32 addr, u32 val) { _MMU_write32<1>(addr & 0xFFFFFFFC, val); }

static const void* mem_funcs[12] =
{
   (void*)_MMU_read08_9 , (void*)_MMU_read08_7,
   (void*)_MMU_write08_9, (void*)_MMU_write08_7,
   (void*)_MMU_read16_9,  (void*)_MMU_read16_7,
   (void*)_MMU_write16_9, (void*)_MMU_write16_7,
   (void*)_MMU_read32_9,  (void*)_MMU_read32_7,
   (void*)_MMU_write32_9, (void*)_MMU_write32_7
};


static OP_RESULT ARM_OP_MEM(uint32_t pc, const uint32_t opcode)
{
   return OPR_INTERPRET;
   const int cond = bit(opcode, 28, 4);
   const bool has_reg_offset = bit(opcode, 25);
   const bool has_pre_index = bit(opcode, 24);
   const bool has_up_bit = bit(opcode, 23);
   const bool has_byte_bit = bit(opcode, 22);
   const bool has_write_back = bit(opcode, 21);
   const bool has_load = bit(opcode, 20);
   const int rd = bit(opcode, 12, 4);
   const int rn = bit(opcode, 16, 4);
   const int rm = bit(opcode, 0, 4); 

   if (rn == 0xf || rd == 0xf || (has_reg_offset && (rm == 0xf)))
      return OPR_INTERPRET;

   const ARM64Reg base = regman.alloc(bit(opcode, 16, 4));
   const ARM64Reg dest = regman.alloc(rd);
   const ARM64Reg offs = has_reg_offset ? regman.alloc(bit(opcode, 0, 4)) : W6;

   auto condBranch = branch_impl(cond);

   // Put the indexed address in R3
   if (has_reg_offset)
   {
      const ShiftType st = (ShiftType)bit(opcode, 5, 2);
      const uint32_t imm = bit(opcode, 7, 5);

      if (has_up_bit) emit.ADD(W6, base, offs, ArithOption(W6, st, imm));
      else            emit.SUB(W6, base, offs, ArithOption(W6, st, imm));
   }
   else
   {
      emit.MOVI2R(W6, opcode & 0xFFF);

      if (has_up_bit) emit.ADD(W6, base, W6);
      else            emit.SUB(W6, base, W6);
   }

   // Load EA
   emit.MOV(W0, has_pre_index ? W6 : base);

   // Do Writeback
   if ((!has_pre_index) || has_write_back)
   {
      emit.MOV(base, W6);
   }

   // DO
   if (!has_load)
   {
      if (has_byte_bit)
      {
         emit.UXTB(W1, dest);
      }
      else
      {
         emit.MOV(W1, dest);
      }
   }

   uint32_t func_idx = block_procnum | (has_load ? 0 : 2) | (has_byte_bit ? 0 : 8);
   call(X7, mem_funcs[func_idx]);
   hw_status_dirty = true;

   if (has_load)
   {
      if (has_byte_bit)
      {
         emit.UXTB(dest, W0);
      }
      else
      {
         emit.MOV(dest, W0);
      }
   }

   // TODO:
   return OPR_RESULT(OPR_CONTINUE, 3);
}

#define ARM_MEM_OP_DEF2(T, Q) \
   static const ArmOpCompiler ARM_OP_##T##_M_LSL_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_P_LSL_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_M_LSR_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_P_LSR_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_M_ASR_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_P_ASR_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_M_ROR_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_P_ROR_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_M_##Q = ARM_OP_MEM; \
   static const ArmOpCompiler ARM_OP_##T##_P_##Q = ARM_OP_MEM

#define ARM_MEM_OP_DEF(T) \
   ARM_MEM_OP_DEF2(T, IMM_OFF_PREIND); \
   ARM_MEM_OP_DEF2(T, IMM_OFF); \
   ARM_MEM_OP_DEF2(T, IMM_OFF_POSTIND)

ARM_MEM_OP_DEF(STR);
ARM_MEM_OP_DEF(LDR);
ARM_MEM_OP_DEF(STRB);
ARM_MEM_OP_DEF(LDRB);

//
static OP_RESULT ARM_OP_MEM_HALF(uint32_t pc, uint32_t opcode)
{
   return OPR_INTERPRET;
   const int cond = bit(opcode, 28, 4);
   const bool has_pre_index = bit(opcode, 24);
   const bool has_up_bit = bit(opcode, 23);
   const bool has_imm_offset = bit(opcode, 22);
   const bool has_write_back = bit(opcode, 21);
   const bool has_load = bit(opcode, 20);
   const uint32_t op = bit(opcode, 5, 2);
   const int rd = bit(opcode, 12, 4);
   const int rn = bit(opcode, 16, 4);
   const int rm = bit(opcode, 0, 4); 

   if (rn == 0xf || rd == 0xf || (!has_imm_offset && (rm == 0xf)))
      return OPR_INTERPRET;

   const ARM64Reg base = regman.alloc(bit(opcode, 16, 4));
   const ARM64Reg dest = regman.alloc(rd);
   const ARM64Reg offs = has_imm_offset ? W0 : regman.alloc(bit(opcode, 0, 4));

   branch_impl(cond);

   // Put the indexed address in R3
   if (!has_imm_offset)
   {
      if (has_up_bit) emit.ADD(W6, base, offs);
      else emit.SUB(W6, base, offs);
   }
   else
   {
      emit.MOVI2R(W6, (opcode & 0xF) | ((opcode >> 4) & 0xF0));

      if (has_up_bit) emit.ADD(W6, base, W6);
      else emit.SUB(W6, base, W6);
   }

   // Load EA
   emit.MOV(W0, has_pre_index ? W6 : base);

   // Do Writeback
   if ((!has_pre_index) || has_write_back)
   {
      emit.MOV(base, W6);
   }

   // DO
   if (!has_load)
   {
      switch (op)
      {
         case 1: emit.UXTH(W1, dest); break;
         case 2: emit.SXTB(W1, dest); break;
         case 3: emit.SXTH(W1, dest); break;
      }
   }

   uint32_t func_idx = block_procnum | (has_load ? 0 : 2) | ((op == 2) ? 0 : 4);
   call(X7, mem_funcs[func_idx]);
   hw_status_dirty = true;
   
   if (has_load)
   {
      switch (op)
      {
         case 1: emit.UXTH(dest, W0); break;
         case 2: emit.SXTB(dest, W0); break;
         case 3: emit.SXTH(dest, W0); break;
      }
   }

   // TODO:
   return OPR_RESULT(OPR_CONTINUE, 3);
}

#define ARM_MEM_HALF_OP_DEF2(T, P) \
   static const ArmOpCompiler ARM_OP_##T##_##P##M_REG_OFF = ARM_OP_MEM_HALF; \
   static const ArmOpCompiler ARM_OP_##T##_##P##P_REG_OFF = ARM_OP_MEM_HALF; \
   static const ArmOpCompiler ARM_OP_##T##_##P##M_IMM_OFF = ARM_OP_MEM_HALF; \
   static const ArmOpCompiler ARM_OP_##T##_##P##P_IMM_OFF = ARM_OP_MEM_HALF

#define ARM_MEM_HALF_OP_DEF(T) \
   ARM_MEM_HALF_OP_DEF2(T, POS_INDE_); \
   ARM_MEM_HALF_OP_DEF2(T, ); \
   ARM_MEM_HALF_OP_DEF2(T, PRE_INDE_)

ARM_MEM_HALF_OP_DEF(STRH);
ARM_MEM_HALF_OP_DEF(LDRH);
ARM_MEM_HALF_OP_DEF(STRSB);
ARM_MEM_HALF_OP_DEF(LDRSB);
ARM_MEM_HALF_OP_DEF(STRSH);
ARM_MEM_HALF_OP_DEF(LDRSH);

//
#define SIGNEXTEND_24(i) (((s32)i<<8)>>8)
/*static OP_RESULT ARM_OP_B_BL(uint32_t pc, uint32_t opcode)
{
   const int cond = bit(opcode, 28, 4);
   const bool has_link = bit(opcode, 24);

   const bool unconditional = cond == 14 || cond == 15;
   uint32_t dest = (pc + 8 + (SIGNEXTEND_24(bit(opcode, 0, 24)) << 2));

   branch_impl(cond);

   if (cond == 14)
   {
      change_mode(true);

      if (has_link)
      {
         dest += 2;
      }
   }

   if (has_link || cond == 15)
   {
      emit.MOVI2R(map_register(14), pc + 4);
   }

   emit.MOVI2R(W6, dest);

   branch_close(cond);

   emit.STR(INDEX_UNSIGNED, W6, RCPU, offsetof(armcpu_t, instruct_adr));

   // TODO: Timing
   return OPR_RESULT(OPR_BRANCHED, 3);
}

#define ARM_OP_B  ARM_OP_B_BL
#define ARM_OP_BL ARM_OP_B_BL*/
#define ARM_OP_B  0
#define ARM_OP_BL 0

////

#define ARM_OP_LDRD_STRD_POST_INDEX 0
#define ARM_OP_LDRD_STRD_OFFSET_PRE_INDEX 0
#define ARM_OP_MRS_CPSR 0
#define ARM_OP_SWP 0
#define ARM_OP_MSR_CPSR 0
#define ARM_OP_BX 0
#define ARM_OP_BLX_REG 0
#define ARM_OP_BKPT 0
#define ARM_OP_MRS_SPSR 0
#define ARM_OP_SWPB 0
#define ARM_OP_MSR_SPSR 0
#define ARM_OP_STREX 0
#define ARM_OP_LDREX 0
#define ARM_OP_MSR_CPSR_IMM_VAL 0
#define ARM_OP_MSR_SPSR_IMM_VAL 0
#define ARM_OP_STMDA 0
#define ARM_OP_LDMDA 0
#define ARM_OP_STMDA_W 0
#define ARM_OP_LDMDA_W 0
#define ARM_OP_STMDA2 0
#define ARM_OP_LDMDA2 0
#define ARM_OP_STMDA2_W 0
#define ARM_OP_LDMDA2_W 0
#define ARM_OP_STMIA 0
#define ARM_OP_LDMIA 0
#define ARM_OP_STMIA_W 0
#define ARM_OP_LDMIA_W 0
#define ARM_OP_STMIA2 0
#define ARM_OP_LDMIA2 0
#define ARM_OP_STMIA2_W 0
#define ARM_OP_LDMIA2_W 0
#define ARM_OP_STMDB 0
#define ARM_OP_LDMDB 0
#define ARM_OP_STMDB_W 0
#define ARM_OP_LDMDB_W 0
#define ARM_OP_STMDB2 0
#define ARM_OP_LDMDB2 0
#define ARM_OP_STMDB2_W 0
#define ARM_OP_LDMDB2_W 0
#define ARM_OP_STMIB 0
#define ARM_OP_LDMIB 0
#define ARM_OP_STMIB_W 0
#define ARM_OP_LDMIB_W 0
#define ARM_OP_STMIB2 0
#define ARM_OP_LDMIB2 0
#define ARM_OP_STMIB2_W 0
#define ARM_OP_LDMIB2_W 0
#define ARM_OP_STC_OPTION 0
#define ARM_OP_LDC_OPTION 0
#define ARM_OP_STC_M_POSTIND 0
#define ARM_OP_LDC_M_POSTIND 0
#define ARM_OP_STC_P_POSTIND 0
#define ARM_OP_LDC_P_POSTIND 0
#define ARM_OP_STC_M_IMM_OFF 0
#define ARM_OP_LDC_M_IMM_OFF 0
#define ARM_OP_STC_M_PREIND 0
#define ARM_OP_LDC_M_PREIND 0
#define ARM_OP_STC_P_IMM_OFF 0
#define ARM_OP_LDC_P_IMM_OFF 0
#define ARM_OP_STC_P_PREIND 0
#define ARM_OP_LDC_P_PREIND 0
#define ARM_OP_CDP 0
#define ARM_OP_MCR 0
#define ARM_OP_MRC 0
#define ARM_OP_SWI 0
#define ARM_OP_UND 0
static const ArmOpCompiler arm_instruction_compilers[4096] = {
#define TABDECL(x) ARM_##x
#include "instruction_tabdef.inc"
#undef TABDECL
};

////////
// THUMB
////////
static OP_RESULT THUMB_OP_SHIFT(uint32_t pc, uint32_t opcode)
{
   return OPR_INTERPRET;
   const ARM64Reg rd = regman.alloc(bit(opcode, 0, 3));
   const ARM64Reg rs = regman.alloc(bit(opcode, 3, 3));
   const uint32_t imm = bit(opcode, 6, 5);
   const int op = bit(opcode, 11, 2);

   emit.MOV(rd, rs, ArithOption(rd, (ShiftType)op, imm));
   emit.TST(rd, rd);
   hw_status_dirty = true;

   return OPR_RESULT(OPR_CONTINUE, 1);
}

#define THUMB_OP_ADDSUB_REGIMM 0
/*static OP_RESULT THUMB_OP_ADDSUB_REGIMM(uint32_t pc, uint32_t opcode)
{
   const uint32_t rd = bit(opcode, 0, 3);
   const uint32_t rs = bit(opcode, 3, 3);
   const AG_ALU_OP op = bit(opcode, 9) ? SUBS : ADDS;
   const bool arg_type = bit(opcode, 10);
   const uint32_t arg = bit(opcode, 6, 3);

   int32_t regs[3] = { rd | 0x10, rs, (!arg_type) ? arg : -1 };
   regman->get(3, regs);

   const reg_t nrd = regs[0];
   const reg_t nrs = regs[1];

   if (arg_type) // Immediate
   {
      block->alu_op(op, nrd, nrs, alu2::imm(arg));
      mark_status_dirty();
   }
   else
   {
      block->alu_op(op, nrd, nrs, alu2::reg(regs[2]));
      mark_status_dirty();
   }

   regman->mark_dirty(nrd);

   return OPR_RESULT(OPR_CONTINUE, 1);
}*/

#define THUMB_OP_MCAS_IMM8 0
/*static OP_RESULT THUMB_OP_MCAS_IMM8(uint32_t pc, uint32_t opcode)
{
   const reg_t rd = bit(opcode, 8, 3);
   const uint32_t op = bit(opcode, 11, 2);
   const uint32_t imm = bit(opcode, 0, 8);

   int32_t regs[1] = { rd };
   regman->get(1, regs);
   const reg_t nrd = regs[0];

   switch (op)
   {
      case 0: block->alu_op(MOVS, nrd, nrd, alu2::imm(imm)); break;
      case 1: block->alu_op(CMP , nrd, nrd, alu2::imm(imm)); break;
      case 2: block->alu_op(ADDS, nrd, nrd, alu2::imm(imm)); break;
      case 3: block->alu_op(SUBS, nrd, nrd, alu2::imm(imm)); break;
   }

   mark_status_dirty();

   if (op != 1) // Don't keep the result of a CMP instruction
   {
      regman->mark_dirty(nrd);
   }

   return OPR_RESULT(OPR_CONTINUE, 1);
}*/

#define THUMB_OP_ALU 0
/*static OP_RESULT THUMB_OP_ALU(uint32_t pc, uint32_t opcode)
{
   const uint32_t rd = bit(opcode, 0, 3);
   const uint32_t rs = bit(opcode, 3, 3);
   const uint32_t op = bit(opcode, 6, 4);
   bool need_writeback = false;

   if (op == 13) // TODO: The MULS is interpreted for now
   {
      return OPR_INTERPRET;
   }

   int32_t regs[2] = { rd, rs };
   regman->get(2, regs);

   const reg_t nrd = regs[0];
   const reg_t nrs = regs[1];

   switch (op)
   {
      case  0: block->ands(nrd, alu2::reg(nrs)); break;
      case  1: block->eors(nrd, alu2::reg(nrs)); break;
      case  5: block->adcs(nrd, alu2::reg(nrs)); break;
      case  6: block->sbcs(nrd, alu2::reg(nrs)); break;
      case  8: block->tst (nrd, alu2::reg(nrs)); break;
      case 10: block->cmp (nrd, alu2::reg(nrs)); break;
      case 11: block->cmn (nrd, alu2::reg(nrs)); break;
      case 12: block->orrs(nrd, alu2::reg(nrs)); break;
      case 14: block->bics(nrd, alu2::reg(nrs)); break;
      case 15: block->mvns(nrd, alu2::reg(nrs)); break;

      case  2: block->movs(nrd, alu2::reg_shift_reg(nrd, LSL, nrs)); break;
      case  3: block->movs(nrd, alu2::reg_shift_reg(nrd, LSR, nrs)); break;
      case  4: block->movs(nrd, alu2::reg_shift_reg(nrd, ASR, nrs)); break;
      case  7: block->movs(nrd, alu2::reg_shift_reg(nrd, arm_gen::ROR, nrs)); break;

      case  9: block->rsbs(nrd, nrs, alu2::imm(0)); break;
   }

   mark_status_dirty();

   static const bool op_wb[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1 };
   if (op_wb[op])
   {
      regman->mark_dirty(nrd);
   }

   return OPR_RESULT(OPR_CONTINUE, 1);
}*/

#define THUMB_OP_SPE 0
/*static OP_RESULT THUMB_OP_SPE(uint32_t pc, uint32_t opcode)
{
   const uint32_t rd = bit(opcode, 0, 3) + (bit(opcode, 7) ? 8 : 0);
   const uint32_t rs = bit(opcode, 3, 4);
   const uint32_t op = bit(opcode, 8, 2);

   if (rd == 0xF || rs == 0xF)
   {
      return OPR_INTERPRET;
   }

   int32_t regs[2] = { rd, rs };
   regman->get(2, regs);

   const reg_t nrd = regs[0];
   const reg_t nrs = regs[1];

   switch (op)
   {
      case 0: block->add(nrd, alu2::reg(nrs)); break;
      case 1: block->cmp(nrd, alu2::reg(nrs)); break;
      case 2: block->mov(nrd, alu2::reg(nrs)); break;
   }

   if (op != 1)
   {
      regman->mark_dirty(nrd);
   }
   else
   {
      mark_status_dirty();
   }

   return OPR_RESULT(OPR_CONTINUE, 1);
}

static OP_RESULT THUMB_OP_MEMORY_DELEGATE(uint32_t pc, uint32_t opcode, bool LOAD, uint32_t SIZE, uint32_t EXTEND, bool REG_OFFSET)
{
   const uint32_t rd = bit(opcode, 0, 3);
   const uint32_t rb = bit(opcode, 3, 3);
   const uint32_t ro = bit(opcode, 6, 3);
   const uint32_t off = bit(opcode, 6, 5);

   int32_t regs[3] = { rd | (LOAD ? 0x10 : 0), rb, REG_OFFSET ? ro : -1};
   regman->get(3, regs);

   const reg_t dest = regs[0];
   const reg_t base = regs[1];

   // Calc EA

   if (REG_OFFSET)
   {
      const reg_t offset = regs[2];
      block->mov(0, alu2::reg(base));
      block->add(0, alu2::reg(offset));
   }
   else
   {
      block->add(0, base, alu2::imm(off << SIZE));
   }

   // Load access function
   block->load_constant(2, mem_funcs[(SIZE << 2) + (LOAD ? 0 : 2) + block_procnum]);

   if (!LOAD)
   {
      block->mov(1, alu2::reg(dest));
   }

   call(2);

   if (LOAD)
   {
      if (EXTEND)
      {
         if (SIZE == 0)
         {
            block->sxtb(dest, 0);
         }
         else
         {
            block->sxth(dest, 0);
         }
      }
      else
      {
         block->mov(dest, alu2::reg(0));
      }

      regman->mark_dirty(dest);
   }

   // TODO
   return OPR_RESULT(OPR_CONTINUE, 3);
}*/

// SIZE: 0=8, 1=16, 2=32
/*template <bool LOAD, uint32_t SIZE, uint32_t EXTEND, bool REG_OFFSET>
static OP_RESULT THUMB_OP_MEMORY(uint32_t pc, uint32_t opcode)
{
   return THUMB_OP_MEMORY_DELEGATE(pc, opcode, LOAD, SIZE, EXTEND, REG_OFFSET);
}*/

#define THUMB_OP_LDR_PCREL 0
/*static OP_RESULT THUMB_OP_LDR_PCREL(uint32_t pc, uint32_t opcode)
{
   const uint32_t offset = bit(opcode, 0, 8);
   const reg_t rd = bit(opcode, 8, 3);

   int32_t regs[1] = { rd | 0x10 };
   regman->get(1, regs);

   const reg_t dest = regs[0];

   block->load_constant(0, ((pc + 4) & ~2) + (offset << 2));
   block->load_constant(2, mem_funcs[8 + block_procnum]);
   call(2);
   block->mov(dest, alu2::reg(0));

   regman->mark_dirty(dest);
   return OPR_RESULT(OPR_CONTINUE, 3);
}*/

#define THUMB_OP_STR_SPREL 0
/*static OP_RESULT THUMB_OP_STR_SPREL(uint32_t pc, uint32_t opcode)
{
   const uint32_t offset = bit(opcode, 0, 8);
   const reg_t rd = bit(opcode, 8, 3);

   int32_t regs[2] = { rd, 13 };
   regman->get(2, regs);

   const reg_t src = regs[0];
   const reg_t base = regs[1];

   block->add(0, base, alu2::imm_rol(offset, 2));
   block->mov(1, alu2::reg(src));
   block->load_constant(2, mem_funcs[10 + block_procnum]);
   call(2);

   return OPR_RESULT(OPR_CONTINUE, 3);
}*/

#define THUMB_OP_LDR_SPREL 0
/*static OP_RESULT THUMB_OP_LDR_SPREL(uint32_t pc, uint32_t opcode)
{
   const uint32_t offset = bit(opcode, 0, 8);
   const reg_t rd = bit(opcode, 8, 3);

   int32_t regs[2] = { rd | 0x10, 13 };
   regman->get(2, regs);

   const reg_t dest = regs[0];
   const reg_t base = regs[1];

   block->add(0, base, alu2::imm_rol(offset, 2));
   block->load_constant(2, mem_funcs[8 + block_procnum]);
   call(2);
   block->mov(dest, alu2::reg(0));

   regman->mark_dirty(dest);
   return OPR_RESULT(OPR_CONTINUE, 3);
}*/

#define THUMB_OP_B_COND 0
/*static OP_RESULT THUMB_OP_B_COND(uint32_t pc, uint32_t opcode)
{
   const AG_COND cond = (AG_COND)bit(opcode, 8, 4);

   block->load_constant(0, pc + 2);
   block->load_constant(0, (pc + 4) + ((u32)((s8)(opcode&0xFF))<<1), cond);
   block->str(0, RCPU, mem2::imm(offsetof(armcpu_t, instruct_adr)));

   block->add(RCYC, alu2::imm(2), cond);

   return OPR_RESULT(OPR_BRANCHED, 1);
}*/

#define THUMB_OP_B_UNCOND 0
/*static OP_RESULT THUMB_OP_B_UNCOND(uint32_t pc, uint32_t opcode)
{
   int32_t offs = (opcode & 0x7FF) | (bit(opcode, 10) ? 0xFFFFF800 : 0);
   block->load_constant(0, pc + 4 + (offs << 1));

   block->str(0, RCPU, mem2::imm(offsetof(armcpu_t, instruct_adr)));

   return OPR_RESULT(OPR_BRANCHED, 3);
}*/

#define THUMB_OP_ADJUST_SP 0
/*static OP_RESULT THUMB_OP_ADJUST_SP(uint32_t pc, uint32_t opcode)
{
   const uint32_t offs = bit(opcode, 0, 7);

   int32_t regs[1] = { 13 };
   regman->get(1, regs);

   const reg_t sp = regs[0];

   if (bit(opcode, 7)) block->sub(sp, alu2::imm_rol(offs, 2));
   else                block->add(sp, alu2::imm_rol(offs, 2));

   regman->mark_dirty(sp);

   return OPR_RESULT(OPR_CONTINUE, 1);
}*/

#define THUMB_OP_ADD_2PC 0
/*static OP_RESULT THUMB_OP_ADD_2PC(uint32_t pc, uint32_t opcode)
{
   const uint32_t offset = bit(opcode, 0, 8);
   const reg_t rd = bit(opcode, 8, 3);

   int32_t regs[1] = { rd | 0x10 };
   regman->get(1, regs);

   const reg_t dest = regs[0];

   block->load_constant(dest, ((pc + 4) & 0xFFFFFFFC) + (offset << 2));
   regman->mark_dirty(dest);

   return OPR_RESULT(OPR_CONTINUE, 1);
}*/

#define THUMB_OP_ADD_2SP 0
/*static OP_RESULT THUMB_OP_ADD_2SP(uint32_t pc, uint32_t opcode)
{
   const uint32_t offset = bit(opcode, 0, 8);
   const reg_t rd = bit(opcode, 8, 3);

   int32_t regs[2] = { 13, rd | 0x10 };
   regman->get(2, regs);

   const reg_t sp = regs[0];
   const reg_t dest = regs[1];

   block->add(dest, sp, alu2::imm_rol(offset, 2));
   regman->mark_dirty(dest);

   return OPR_RESULT(OPR_CONTINUE, 1);
}*/

#define THUMB_OP_BX_BLX_THUMB 0
/*static OP_RESULT THUMB_OP_BX_BLX_THUMB(uint32_t pc, uint32_t opcode)
{
   const reg_t rm = bit(opcode, 3, 4);
   const bool link = bit(opcode, 7);

   if (rm == 15)
      return OPR_INTERPRET;

   block->load_constant(0, pc + 4);

   int32_t regs[2] = { link ? 14 : -1, (rm != 15) ? (int32_t)rm : -1 };
   regman->get(2, regs);

   if (link)
   {
      const reg_t lr = regs[0];
      block->sub(lr, 0, alu2::imm(1));
      regman->mark_dirty(lr);
   }

   reg_t target = regs[1];

   change_mode_reg(target, 2, 3);
   block->bic(0, target, alu2::imm(1));
   block->str(0, RCPU, mem2::imm(offsetof(armcpu_t, instruct_adr)));

   return OPR_RESULT(OPR_BRANCHED, 3);
}*/

#if 1
#define THUMB_OP_BL_LONG 0
#else
static OP_RESULT THUMB_OP_BL_LONG(uint32_t pc, uint32_t opcode)
{
   static const uint32_t op = bit(opcode, 11, 5);
   int32_t offset = bit(opcode, 0, 11);

   reg_t lr = regman->get(14, op == 0x1E);

   if (op == 0x1E)
   {
      offset |= (offset & 0x400) ? 0xFFFFF800 : 0;
      block->load_constant(lr, (pc + 4) + (offset << 12));
   }
   else
   {
      block->load_constant(0, offset << 1);

      block->add(0, lr, alu2::reg(0));
      block->str(0, RCPU, mem2::imm(offsetof(armcpu_t, instruct_adr)));

      block->load_constant(lr, pc + 3);

      if (op != 0x1F)
      {
         change_mode(false);
      }
   }

   regman->mark_dirty(lr);

   if (op == 0x1E)
   {
      return OPR_RESULT(OPR_CONTINUE, 1);
   }
   else
   {
      return OPR_RESULT(OPR_BRANCHED, (op == 0x1F) ? 3 : 4);
   }
}
#endif

#define THUMB_OP_INTERPRET       0
#define THUMB_OP_UND_THUMB       THUMB_OP_INTERPRET

#define THUMB_OP_LSL             THUMB_OP_SHIFT
#define THUMB_OP_LSL_0           THUMB_OP_SHIFT
#define THUMB_OP_LSR             THUMB_OP_SHIFT
#define THUMB_OP_LSR_0           THUMB_OP_SHIFT
#define THUMB_OP_ASR             THUMB_OP_SHIFT
#define THUMB_OP_ASR_0           THUMB_OP_SHIFT

#define THUMB_OP_ADD_REG         THUMB_OP_ADDSUB_REGIMM
#define THUMB_OP_SUB_REG         THUMB_OP_ADDSUB_REGIMM
#define THUMB_OP_ADD_IMM3        THUMB_OP_ADDSUB_REGIMM
#define THUMB_OP_SUB_IMM3        THUMB_OP_ADDSUB_REGIMM

#define THUMB_OP_MOV_IMM8        THUMB_OP_MCAS_IMM8
#define THUMB_OP_CMP_IMM8        THUMB_OP_MCAS_IMM8
#define THUMB_OP_ADD_IMM8        THUMB_OP_MCAS_IMM8
#define THUMB_OP_SUB_IMM8        THUMB_OP_MCAS_IMM8

#define THUMB_OP_AND             THUMB_OP_ALU
#define THUMB_OP_EOR             THUMB_OP_ALU
#define THUMB_OP_LSL_REG         THUMB_OP_ALU
#define THUMB_OP_LSR_REG         THUMB_OP_ALU
#define THUMB_OP_ASR_REG         THUMB_OP_ALU
#define THUMB_OP_ADC_REG         THUMB_OP_ALU
#define THUMB_OP_SBC_REG         THUMB_OP_ALU
#define THUMB_OP_ROR_REG         THUMB_OP_ALU
#define THUMB_OP_TST             THUMB_OP_ALU
#define THUMB_OP_NEG             THUMB_OP_ALU
#define THUMB_OP_CMP             THUMB_OP_ALU
#define THUMB_OP_CMN             THUMB_OP_ALU
#define THUMB_OP_ORR             THUMB_OP_ALU
#define THUMB_OP_MUL_REG         THUMB_OP_INTERPRET
#define THUMB_OP_BIC             THUMB_OP_ALU
#define THUMB_OP_MVN             THUMB_OP_ALU

#define THUMB_OP_ADD_SPE         THUMB_OP_SPE
#define THUMB_OP_CMP_SPE         THUMB_OP_SPE
#define THUMB_OP_MOV_SPE         THUMB_OP_SPE

#define THUMB_OP_ADJUST_P_SP     THUMB_OP_ADJUST_SP
#define THUMB_OP_ADJUST_M_SP     THUMB_OP_ADJUST_SP
/*
#define THUMB_OP_LDRB_REG_OFF    THUMB_OP_MEMORY<true , 0, 0, true>
#define THUMB_OP_LDRH_REG_OFF    THUMB_OP_MEMORY<true , 1, 0, true>
#define THUMB_OP_LDR_REG_OFF     THUMB_OP_MEMORY<true , 2, 0, true>

#define THUMB_OP_STRB_REG_OFF    THUMB_OP_MEMORY<false, 0, 0, true>
#define THUMB_OP_STRH_REG_OFF    THUMB_OP_MEMORY<false, 1, 0, true>
#define THUMB_OP_STR_REG_OFF     THUMB_OP_MEMORY<false, 2, 0, true>

#define THUMB_OP_LDRB_IMM_OFF    THUMB_OP_MEMORY<true , 0, 0, false>
#define THUMB_OP_LDRH_IMM_OFF    THUMB_OP_MEMORY<true , 1, 0, false>
#define THUMB_OP_LDR_IMM_OFF     THUMB_OP_MEMORY<true , 2, 0, false>

#define THUMB_OP_STRB_IMM_OFF    THUMB_OP_MEMORY<false, 0, 0, false>
#define THUMB_OP_STRH_IMM_OFF    THUMB_OP_MEMORY<false, 1, 0, false>
#define THUMB_OP_STR_IMM_OFF     THUMB_OP_MEMORY<false, 2, 0, false>

#define THUMB_OP_LDRSB_REG_OFF   THUMB_OP_MEMORY<true , 0, 1, true>
#define THUMB_OP_LDRSH_REG_OFF   THUMB_OP_MEMORY<true , 1, 1, true>*/
#define THUMB_OP_LDRB_REG_OFF    0
#define THUMB_OP_LDRH_REG_OFF    0
#define THUMB_OP_LDR_REG_OFF     0

#define THUMB_OP_STRB_REG_OFF    0
#define THUMB_OP_STRH_REG_OFF    0
#define THUMB_OP_STR_REG_OFF     0

#define THUMB_OP_LDRB_IMM_OFF    0
#define THUMB_OP_LDRH_IMM_OFF    0
#define THUMB_OP_LDR_IMM_OFF     0

#define THUMB_OP_STRB_IMM_OFF    0
#define THUMB_OP_STRH_IMM_OFF    0
#define THUMB_OP_STR_IMM_OFF     0

#define THUMB_OP_LDRSB_REG_OFF   0
#define THUMB_OP_LDRSH_REG_OFF   0

#define THUMB_OP_BX_THUMB        THUMB_OP_BX_BLX_THUMB
#define THUMB_OP_BLX_THUMB       THUMB_OP_BX_BLX_THUMB
#define THUMB_OP_BL_10           THUMB_OP_BL_LONG
#define THUMB_OP_BL_11           THUMB_OP_BL_LONG
#define THUMB_OP_BLX             THUMB_OP_BL_LONG


// UNDEFINED OPS
#define THUMB_OP_PUSH            THUMB_OP_INTERPRET
#define THUMB_OP_PUSH_LR         THUMB_OP_INTERPRET
#define THUMB_OP_POP             THUMB_OP_INTERPRET
#define THUMB_OP_POP_PC          THUMB_OP_INTERPRET
#define THUMB_OP_BKPT_THUMB      THUMB_OP_INTERPRET
#define THUMB_OP_STMIA_THUMB     THUMB_OP_INTERPRET
#define THUMB_OP_LDMIA_THUMB     THUMB_OP_INTERPRET
#define THUMB_OP_SWI_THUMB       THUMB_OP_INTERPRET

static const ArmOpCompiler thumb_instruction_compilers[1024] = {
#define TABDECL(x) THUMB_##x
#include "thumb_tabdef.inc"
#undef TABDECL
};



// ============================================================================================= IMM

//-----------------------------------------------------------------------------
//   Compiler
//-----------------------------------------------------------------------------

static u32 instr_attributes(bool thumb, u32 opcode)
{
   return thumb ? thumb_attributes[opcode>>6]
                : instruction_attributes[INSTRUCTION_INDEX(opcode)];
}

static bool instr_is_branch(bool thumb, u32 opcode)
{
   u32 x = instr_attributes(thumb, opcode);
   if(thumb)
      return (x & BRANCH_ALWAYS)
          || ((x & BRANCH_POS0) && ((opcode&7) | ((opcode>>4)&8)) == 15)
          || (x & BRANCH_SWI)
          || (x & JIT_BYPASS);
   else
      return (x & BRANCH_ALWAYS)
          || ((x & BRANCH_POS12) && REG_POS(opcode,12) == 15)
          || ((x & BRANCH_LDM) && bit(opcode, 31))
          || (x & BRANCH_SWI)
          || (x & JIT_BYPASS);
}

static void cyclePrinter(u32 cycles)
{
   printf("cycles %d\n", cycles);
}

template<int PROCNUM>
static ArmOpCompiled compile_basicblock()
{
   assert(jitUsed != -1);
   jitTransitionToWritable(&jitPage);

   block_procnum = PROCNUM;

   const u32 base = ARMPROC.instruct_adr;
   const bool thumb = ARMPROC.CPSR.bits.T == 1;
   const u32 isize = thumb ? 2 : 4;

   uint32_t pc = base;
   bool compiled_op = true;
   bool has_ended = false;
   uint32_t constant_cycles = 0;

   emit.ABI_PushRegisters({30,
      19, 20, 21, 22, 23, 24, 25, 26, 27, 28});
   
   // NOTE: Expected register usage
   // R5 = Pointer to ARMPROC
   // R6 = Cycle counter

   emit.MOVP2R(RCPU, &ARMPROC);
   emit.MOVZ(RCYC, 0);

   regman.reset();
   load_status_from_mem();

   for (uint32_t i = 0; i < CommonSettings.jit_max_block_size && !has_ended; i ++, pc += isize)
   {
      uint32_t opcode = thumb ? _MMU_read16<PROCNUM, MMU_AT_CODE>(pc) : _MMU_read32<PROCNUM, MMU_AT_CODE>(pc);

      ArmOpCompiler compiler = thumb ? thumb_instruction_compilers[opcode >> 6]
                                     : arm_instruction_compilers[INSTRUCTION_INDEX(opcode)];

      int result = compiler ? compiler(pc, opcode) : OPR_INTERPRET;

      constant_cycles += OPR_RESULT_CYCLES(result);
      switch (OPR_RESULT_ACTION(result))
      {
         case OPR_INTERPRET:
         {
            if (compiled_op)
            {
               arm_jit_prefetch<PROCNUM>(pc, opcode, thumb);
               compiled_op = false;
            }

            regman.flush();
            regman.reset();
            write_status_to_mem();

            call(X6, armcpu_exec<PROCNUM>);
            emit.ADD(RCYC, RCYC, W0);

            load_status_from_mem();

            has_ended = has_ended || instr_is_branch(thumb, opcode);

            break;
         }

         case OPR_BRANCHED:
         {
            has_ended = true;
            compiled_op = false;
            break;
         }

         case OPR_CONTINUE:
         {
            compiled_op = true;
            break;
         }
      }
   }

   if (compiled_op)
   {
      emit.MOVI2R(W6, pc);
      emit.STR(INDEX_UNSIGNED, W6, RCPU, offsetof(armcpu_t, instruct_adr));
   }

   regman.flush();
   write_status_to_mem();

   emit.MOVI2R(W6, constant_cycles);
   emit.ADD(W0, W6, RCYC);

   emit.ABI_PopRegisters({30,
      19, 20, 21, 22, 23, 24, 25, 26, 27, 28});
   emit.RET();

   /*FILE* f = fopen("instruction dump.txt", "a");
   fprintf(f, "instruction offset %x\n", jitUsed);
   u32* instrPtr = (u32*)(jitRWAddr + jitUsed);
   while(instrPtr != (u32*)emit.GetCodePtr())
      fprintf(f, "%x\n", *(instrPtr++));
   fclose(f);*/

   void* fn_ptr = jitRXAddr + jitUsed;
   //printf("compiled block %d %p\n", jitUsed, fn_ptr);
   JIT_COMPILED_FUNC(base, PROCNUM) = (uintptr_t)fn_ptr;

   jitTransitionToExecutable(&jitPage);

   jitUsed = emit.GetCodePtr() - jitRWAddr;
   //printf("advanced to %d %p\n", emit.GetCodePtr() - jitRWAddr, jitRXAddr);

   return (ArmOpCompiled)fn_ptr;
}


template<int PROCNUM> u32 arm_jit_compile()
{
   u32 adr = ARMPROC.instruct_adr;
   u32 mask_adr = (adr & 0x07FFFFFE) >> 4;
   if(((recompile_counts[mask_adr >> 1] >> 4*(mask_adr & 1)) & 0xF) > 8)
   {
      ArmOpCompiled f = op_decode[PROCNUM][ARMPROC.CPSR.bits.T];
      JIT_COMPILED_FUNC(adr, PROCNUM) = (uintptr_t)f;
      return f();
   }

   recompile_counts[mask_adr >> 1] += 1 << 4*(mask_adr & 1);

   if ((JitPageSize - jitUsed) / sizeof(uint32_t) < 1000)
   {
      arm_jit_reset(true);
   }

   //printf("compiling\n");
   auto resFunc = compile_basicblock<PROCNUM>();
   //printf("executing\n");
   auto result = resFunc();
   //printf("finished\n");
   return result;
}

template u32 arm_jit_compile<0>();
template u32 arm_jit_compile<1>();

void arm_jit_reset(bool enable, bool suppress_msg)
{
   if (!suppress_msg)
	   printf("CPU mode: %s\n", enable?"JIT":"Interpreter");

   saveBlockSizeJIT = CommonSettings.jit_max_block_size;

   if (enable)
   {
      printf("JIT: max block size %d instruction(s)\n", CommonSettings.jit_max_block_size);

#ifdef MAPPED_JIT_FUNCS

      #define JITFREE(x) memset(x,0,sizeof(x));
         JITFREE(JIT.MAIN_MEM);
         JITFREE(JIT.SWIRAM);
         JITFREE(JIT.ARM9_ITCM);
         JITFREE(JIT.ARM9_LCDC);
         JITFREE(JIT.ARM9_BIOS);
         JITFREE(JIT.ARM7_BIOS);
         JITFREE(JIT.ARM7_ERAM);
         JITFREE(JIT.ARM7_WIRAM);
         JITFREE(JIT.ARM7_WRAM);
      #undef JITFREE

      memset(recompile_counts, 0, sizeof(recompile_counts));
      init_jit_mem();

#else
      for(int i=0; i<sizeof(recompile_counts)/8; i++)
         if(((u64*)recompile_counts)[i])
         {
            ((u64*)recompile_counts)[i] = 0;
            memset(compiled_funcs+128*i, 0, 128*sizeof(*compiled_funcs));
         }
#endif

      if(jitUsed == -1) {
         jitCreate(&jitPage, JitPageSize);
         jitRWAddr = (u8*)jitGetRwAddr(&jitPage);
         jitRXAddr = (u8*)jitGetRxAddr(&jitPage);
         jitTransitionToExecutable(&jitPage);
      }
      jitUsed = 0;
      emit.SetCodePtr(jitRWAddr);
   }
}

void arm_jit_close()
{
   if(jitUsed != -1)
      jitClose(&jitPage);
}
#endif // HAVE_JIT
