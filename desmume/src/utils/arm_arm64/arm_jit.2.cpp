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
static const int JitPageSize = INSTRUCTION_COUNT * 4;

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
   u8 regs_locked;

   int stashed_pairs[4];
   int stashed_pairs_count = 0;
   int pairs_stashed = 0;

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
         if (reg_usage[i] < usage && !(regs_locked & (1 << i))) {
            usage = reg_usage[i];
            reg = i;
         }
      }
      assert(reg != -1 && "couldn't allocate new register");

      return reg;
   }
public:
   reg_manager()
   {
      reset();
   }

   void pop_stashed_regs()
   {
      /*for (int i = stashed_pairs_count - 1; i >= 0; i--) {
         emit.ABI_PopRegisters({19 + stashed_pairs[i], 19 + stashed_pairs[i] + 1});
         //printf("popping %d %d\n", stashed_list[n], stashed_list[n] + 1);
      }
      stashed_pairs_count = 0;
      pairs_stashed = 0;*/
   }

   void reset()
   {
      //printf("reset\n");
      for (int i = 0; i < 8; i++)
         emu_regs[i] = reg_usage[i] = -1;
      for (int i = 0; i < 16; i++)
         native_regs[i] = -1;
      
      regs_dirty = 0;
      regs_locked = 0;
   }

   void flush()
   {
      for (int i = 0; i < 8; i++)
         if (regs_dirty & (1 << i))
         {
            //printf("str native %d -> emu %d (flush)\n", get_reg_n(i), emu_regs[i]);
            emit.STR(INDEX_UNSIGNED, get_reg_n(i), RCPU, offsetof(armcpu_t, R) + 4 * emu_regs[i]);
         }
      regs_dirty = 0;
   }

   void unlock()
   {
      regs_locked = 0;
   }

   ARM64Reg alloc(int reg, bool mut = false)
   {
      int native_reg = -1;
      if (native_regs[reg] != -1)
         native_reg = native_regs[reg];
      else
      {
         native_reg = get_least_used();

         /*int stash_pair = native_reg & ~1;
         int stash_bit = 1 << (stash_pair / 2);
         if (!(pairs_stashed & stash_bit)) {
            emit.ABI_PushRegisters({19 + stash_pair, 19 + stash_pair + 1});
            //printf("pushing %d %d\n", stash_pair, stash_pair + 1);
            stashed_pairs[stashed_pairs_count++] = stash_pair;
            pairs_stashed |= stash_bit;
         }*/

         if (emu_regs[native_reg] != -1) {
            if (regs_dirty & (1 << native_reg)) {
               //printf("str native %d -> emu %d\n", get_reg_n(native_reg), emu_regs[native_reg]);
               emit.STR(INDEX_UNSIGNED, get_reg_n(native_reg), RCPU, offsetof(armcpu_t, R) + 4 * emu_regs[native_reg]);
               regs_dirty &= ~(1 << native_reg);
            }
            native_regs[emu_regs[native_reg]] = -1;
         }
         
         //printf("ldr native %d <- emu %d (mut: %d)\n", get_reg_n(native_reg), reg, mut);
         emit.LDR(INDEX_UNSIGNED, get_reg_n(native_reg), RCPU, offsetof(armcpu_t, R) + 4 * reg);

         native_regs[reg] = native_reg;
         emu_regs[native_reg] = reg;
         reg_usage[native_reg] = 0;
      }

      regs_dirty |= (mut << native_reg);
      regs_locked |= (1 << native_reg);

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
   if (branch)
   {
      //printf("skipping instruction\n");
      emit.SetJumpTarget(*branch);
   }
}

static std::optional<FixupBranch> branch_impl(int cond)
{
   if(cond != 0xe) {
      flush_hwstatus();
      return emit.B((CCFlags)((u32)cond ^ 1));
      /*hw_status_dirty = true;
      // TODO: könnte verkompaktiert werden
      emit.UBFX(W9, RCPSR, 24, 8);
      if(cond < 8)
      {
         static const u8 cond_bit[] = {0x40, 0x40, 0x20, 0x20, 0x80, 0x80, 0x10, 0x10};
         emit.TSTI2R(W9, cond_bit[cond], W10);
         return (cond & 1) ? emit.B(CC_NEQ) : emit.B(CC_EQ);
      }
      else
      {
         emit.ANDI2R(W9, W9, 0xf0, W10);
         emit.MOVP2R(W10, arm_cond_table);
         emit.ADD(W10, W10, W9);
         emit.LDURB(W10, W10, cond);
         return emit.CBZ(W10);
      }*/
   }
   return std::optional<FixupBranch>();
}

static void change_mode(bool thumb)
{
   if (!thumb)
   {
      emit.ANDI2R(RCPSR, RCPSR, 0x20);
   }
   else
   {
      emit.ORRI2R(RCPSR, RCPSR, 0x20);
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

   emit.MOVI2R(W9, pc & imask);
   emit.STR(INDEX_UNSIGNED, W9, RCPU, offsetof(armcpu_t, instruct_adr));
   
   emit.ADD(W9, W9, isize);
   emit.STR(INDEX_UNSIGNED, W9, RCPU, offsetof(armcpu_t, next_instruction));

   emit.ADD(W9, W9, isize);
   emit.STR(INDEX_UNSIGNED, W9, RCPU, offsetof(armcpu_t, R) + 4 * 15);

   emit.MOVI2R(W9, opcode);
   emit.STR(INDEX_UNSIGNED, W9, RCPU, offsetof(armcpu_t, instruction));
}

/////////
/// ARM
/////////

/*
      Instruktionen wo rhs Shift um Immediate ist
            => direkt übersetzen

*/
// welcome to pre processor hell…

#define use(x, y) x
#define use_not(x, y) y

#define ARM_ALU_OP_DEF(name, body, cyc, has_rd, has_rn, has_rm, has_rs, has_shiftimm, has_immval) \
   static OP_RESULT ARM_OP_##name(uint32_t pc, uint32_t opcode) { \
      has_rd( \
         const int rd = bit(opcode, 12, 4); \
         if (rd == 0xf) return OPR_INTERPRET; \
         const ARM64Reg rd_native = regman.alloc(rd, true); \
      ,) \
      has_rn( \
         const int rn = bit(opcode, 16, 4); \
         if (rn == 0xf) return OPR_INTERPRET; \
         const ARM64Reg rn_native = regman.alloc(rn); \
      ,) \
      has_rm( \
         const int rm = bit(opcode, 0, 4); \
         if (rm == 0xf) return OPR_INTERPRET; \
         const ARM64Reg rm_native = regman.alloc(rm); \
      ,) \
      has_rs( \
         const int rs = bit(opcode, 8, 4); \
         const ARM64Reg rs_native = regman.alloc(rs); \
      ,) \
      has_shiftimm( \
         const int shift_amount = bit(opcode, 7, 5); \
         if (bit(opcode, 5, 2) != 0 && shift_amount == 0) return OPR_INTERPRET; \
      ,) \
      has_immval( \
         const int imm_val = bit(opcode, 0, 8); \
         const int imm_ror = bit(opcode, 8, 4); \
      ,) \
      const int cond = bit(opcode, 28, 4); \
      auto condBranch = branch_impl(cond); \
      body \
      close_branch(condBranch); \
      regman.unlock(); \
      \
      return OPR_RESULT(OPR_CONTINUE, cyc); \
   }

#define LSL_IMM_S \
   if (shift_amount != 0) \
   { \
      emit.UBFX(W9, rm_native, 32 - shift_amount, 1); \
      emit.BFI(RCPSR, W9, FCarryShift, 1); \
   }
#define LSR_IMM_S \
   if (shift_amount != 0) \
      emit.UBFX(W9, rm_native, shift_amount - 1, 1); \
   else \
      emit.UBFX(W9, rm_native, 31, 1); \
   emit.BFI(RCPSR, W9, FCarryShift, 1);
#define ASR_IMM_S LSR_IMM_S
#define ROR_IMM_S \
   if (shift_amount != 0) \
      emit.UBFX(W9, rm_native, shift_amount - 1, 1); \
   else \
      emit.UBFX(W9, rm_native, 0, 1); \
   emit.BFI(RCPSR, W9, FCarryShift, 1);
#define IMM_VAL_S \
   if (imm_ror != 0) \
   { \
      emit.MOVZ(W9, bit(imm_val, imm_ror * 2 - 1)); \
      emit.BFI(RCPSR, W9, FCarryShift, 1); \
   }

#define DEF_ARITHMETIC(name, op, arg) \
   ARM_ALU_OP_DEF(name##_LSL_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSL, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_LSR_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSR, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_ASR_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_ASR, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_ROR_IMM, emit.ROR(W9, rm_native, shift_amount); emit.op(rd_native, rn_native, W9);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_IMM_VAL, \
      emit.MOVI2R(W9, ::ROR(imm_val, imm_ror * 2)); emit.op(rd_native, rn_native, W9);, \
      2, use, use, use_not, use_not, use_not, use) \
   \
   ARM_ALU_OP_DEF(name##_S_LSL_IMM, \
      return OPR_INTERPRET; emit.op##S(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSL, shift_amount)); \
      copy_hwstatus(true, true, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_LSR_IMM, \
      emit.op##S(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSR, shift_amount)); \
      copy_hwstatus(true, true, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_ASR_IMM, \
      emit.op##S(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_ASR, shift_amount)); \
      copy_hwstatus(true, true, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_ROR_IMM, \
      emit.ROR(W9, rm_native, shift_amount); emit.op##S(rd_native, rn_native, W9); \
      copy_hwstatus(true, true, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_IMM_VAL, \
      emit.MOVI2R(W9, ::ROR(imm_val, imm_ror * 2)); emit.op##S(rd_native, rn_native, W9); \
      copy_hwstatus(true, true, true, true);, \
      2, use, use, use_not, use_not, use_not, use)

/*DEF_ARITHMETIC(SUB, SUB, )
DEF_ARITHMETIC(ADD, ADD, )*/

#define DEF_LOGICAL(name, op) \
   ARM_ALU_OP_DEF(name##_LSL_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSL, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_LSR_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSR, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_ASR_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_ASR, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_ROR_IMM, emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_ROR, shift_amount));, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_IMM_VAL, \
      emit.MOVI2R(W9, ::ROR(imm_val, imm_ror * 2)); emit.op(rd_native, rn_native, W9);, \
      2, use, use, use_not, use_not, use_not, use) \
   \
   ARM_ALU_OP_DEF(name##_S_LSL_IMM, LSL_IMM_S \
      emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSL, shift_amount)); \
      emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_LSR_IMM, LSR_IMM_S \
      emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_LSR, shift_amount)); \
      emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_ASR_IMM, ASR_IMM_S \
      emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_ASR, shift_amount)); \
      emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_ROR_IMM, ROR_IMM_S \
      emit.op(rd_native, rn_native, rm_native, ArithOption(rd_native, ST_ROR, shift_amount)); \
      emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, \
      1, use, use, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_S_IMM_VAL, IMM_VAL_S \
      emit.MOVI2R(W9, ::ROR(imm_val, imm_ror * 2)); emit.op(rd_native, rn_native, W9); \
      emit.TST(rd_native, rd_native); copy_hwstatus(false, false, true, true);, \
      2, use, use, use_not, use_not, use_not, use)

/*DEF_LOGICAL(AND, AND)
DEF_LOGICAL(EOR, EOR)
DEF_LOGICAL(ORR, ORR)
DEF_LOGICAL(BIC, BIC)*/

#define DEF_MOV_DIRECTTRANS(name, op) \
   ARM_ALU_OP_DEF(name##_LSL_IMM, emit.op(rd_native, WZR, rm_native, ArithOption(rd_native, ST_LSL, shift_amount));, \
      1, use, use_not, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_LSR_IMM, emit.op(rd_native, WZR, rm_native, ArithOption(rd_native, ST_LSR, shift_amount));, \
      1, use, use_not, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_ASR_IMM, emit.op(rd_native, WZR, rm_native, ArithOption(rd_native, ST_ASR, shift_amount));, \
      1, use, use_not, use, use_not, use, use_not) \
   ARM_ALU_OP_DEF(name##_ROR_IMM, emit.op(rd_native, WZR, rm_native, ArithOption(rd_native, ST_ROR, shift_amount));, \
      1, use, use_not, use, use_not, use, use_not) \
   /*ARM_ALU_OP_DEF(name##_IMM_VAL, \
      emit.MOVI2R(W9, ::ROR(imm_val, imm_ror * 2)); emit.op(rd_native, WZR, W9);, \
      2, use, use_not, use_not, use_not, use_not, use)*/

DEF_MOV_DIRECTTRANS(MOV, ORR)
DEF_MOV_DIRECTTRANS(MVN, ORN)

#define ARM_ALU_OP_DEF_NULL(T, D, N, S) \
   /*static const ArmOpCompiler ARM_OP_##T##_LSL_IMM = nullptr;*/ \
   static const ArmOpCompiler ARM_OP_##T##_LSL_REG = nullptr; \
   /*static const ArmOpCompiler ARM_OP_##T##_LSR_IMM = nullptr;*/ \
   static const ArmOpCompiler ARM_OP_##T##_LSR_REG = nullptr; \
   /*static const ArmOpCompiler ARM_OP_##T##_ASR_IMM = nullptr;*/ \
   static const ArmOpCompiler ARM_OP_##T##_ASR_REG = nullptr; \
   /*static const ArmOpCompiler ARM_OP_##T##_ROR_IMM = nullptr;*/ \
   static const ArmOpCompiler ARM_OP_##T##_ROR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_IMM_VAL = nullptr;
#define ARM_ALU_OP_DEF_ALL_NULL(T, D, N, S) \
   static const ArmOpCompiler ARM_OP_##T##_LSL_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_LSL_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_LSR_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_LSR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ASR_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ASR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ROR_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ROR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_IMM_VAL = nullptr

ARM_ALU_OP_DEF_ALL_NULL(AND  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(AND_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(EOR  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(EOR_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(SUB  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(SUB_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(RSB  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(RSB_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(ADD  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(ADD_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(ADC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(ADC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(SBC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(SBC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(RSC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(RSC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(TST  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(TEQ  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(CMP  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(CMN  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(ORR  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(ORR_S, 2, 1, true);
ARM_ALU_OP_DEF_NULL(MOV  , 2, 0, false);
ARM_ALU_OP_DEF_ALL_NULL(MOV_S, 2, 0, true);
ARM_ALU_OP_DEF_ALL_NULL(BIC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(BIC_S, 2, 1, true);
ARM_ALU_OP_DEF_NULL(MVN  , 2, 0, false);
ARM_ALU_OP_DEF_ALL_NULL(MVN_S, 2, 0, true);

#define ARM_OP_MUL         0
#define ARM_OP_MUL_S       0
#define ARM_OP_MLA         0
#define ARM_OP_MLA_S       0
#define ARM_OP_UMULL       0
#define ARM_OP_UMULL_S     0
#define ARM_OP_UMLAL       0
#define ARM_OP_UMLAL_S     0
#define ARM_OP_SMULL       0
#define ARM_OP_SMULL_S     0
#define ARM_OP_SMLAL       0
#define ARM_OP_SMLAL_S     0

#define ARM_OP_SMUL_B_B    0
#define ARM_OP_SMUL_T_B    0
#define ARM_OP_SMUL_B_T    0
#define ARM_OP_SMUL_T_T    0

#define ARM_OP_SMLA_B_B    0
#define ARM_OP_SMLA_T_B    0
#define ARM_OP_SMLA_B_T    0
#define ARM_OP_SMLA_T_T    0

#define ARM_OP_SMULW_B     0
#define ARM_OP_SMULW_T     0
#define ARM_OP_SMLAW_B     0
#define ARM_OP_SMLAW_T     0

#define ARM_OP_SMLAL_B_B   0
#define ARM_OP_SMLAL_T_B   0
#define ARM_OP_SMLAL_B_T   0
#define ARM_OP_SMLAL_T_T   0

#define ARM_OP_QADD        0
#define ARM_OP_QSUB        0
#define ARM_OP_QDADD       0
#define ARM_OP_QDSUB       0

#define ARM_OP_CLZ         0

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
static OP_RESULT ARM_OP_B_BL(uint32_t pc, uint32_t opcode)
{
   const int cond = bit(opcode, 28, 4);
   const bool has_link = bit(opcode, 24);

   const bool unconditional = cond == 14 || cond == 15;
   uint32_t dest = (pc + 8 + (SIGNEXTEND_24(bit(opcode, 0, 24)) << 2));

   auto branch = branch_impl(cond);

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
      const ARM64Reg r14 = regman.alloc(14, true);
      emit.MOVI2R(r14, pc + 4);
   }

   emit.MOVI2R(W6, dest);
   emit.STR(INDEX_UNSIGNED, W6, RCPU, offsetof(armcpu_t, instruct_adr));

   close_branch(branch);

   // TODO: Timing
   return OPR_RESULT(OPR_BRANCHED, 3);
}

#define ARM_OP_B  ARM_OP_B_BL
#define ARM_OP_BL ARM_OP_B_BL
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
#define THUMB_OP_SHIFT 0
/*static OP_RESULT THUMB_OP_SHIFT(uint32_t pc, uint32_t opcode)
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
}*/

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

static bool instr_does_prefetch(bool thumb, u32 opcode)
{
	u32 x = instr_attributes(thumb, opcode);
	if(thumb)
		return thumb_instruction_compilers[opcode>>6]
			   && (x & BRANCH_ALWAYS);
	else
		return instr_is_branch(thumb, opcode) && arm_instruction_compilers[INSTRUCTION_INDEX(opcode)]
			   && ((x & BRANCH_ALWAYS) || (x & BRANCH_LDM));
}

static bool instr_is_conditional(bool thumb, u32 opcode)
{
	if(thumb) return false;
	
	return !(CONDITION(opcode) == 0xE
	         || (CONDITION(opcode) == 0xF && CODE(opcode) == 5));
}

static void printRegs(u32 opcode, armcpu_t* cpu, u32 interpret)
{
   printf("%d %x R: ", interpret, opcode);
   for (int i = 0; i < 16; i++)
      printf("%x ", cpu->R[i]);

   printf(" CPSR: %x\n", cpu->CPSR);
}
static void printStart(u32 pc, u32 thumb, u32 cpsr)
{
   printf("block start %x %x %x\n", pc, thumb, cpsr);
}

static int instructionsTotal = 0;
static int instructionsInterpreted = 0;
static int instructionsThumb = 0;

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

   emit.ABI_PushRegisters({30, 28, 19, 20, 21, 22, 23, 24, 25, 26, 27}); // link and CPSR
   
   // NOTE: Expected register usage
   // R5 = Pointer to ARMPROC
   // R6 = Cycle counter

   emit.MOVP2R(RCPU, &ARMPROC);
   emit.MOVZ(RCYC, 0);

   regman.reset();
   load_status_from_mem();

   bool printEverything = true;

   uint32_t opcode = 0;

   for (uint32_t i = 0; i < CommonSettings.jit_max_block_size && !has_ended; i ++, pc += isize)
   {
      opcode = thumb ? _MMU_read16<PROCNUM, MMU_AT_CODE>(pc) : _MMU_read32<PROCNUM, MMU_AT_CODE>(pc);

      ArmOpCompiler compiler = thumb ? thumb_instruction_compilers[opcode >> 6]
                                     : arm_instruction_compilers[INSTRUCTION_INDEX(opcode)];

      int result = compiler ? compiler(pc, opcode) : OPR_INTERPRET;

      instructionsTotal++;
      instructionsThumb += thumb;

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

            flush_hwstatus();

            call(X6, armcpu_exec<PROCNUM>);
            emit.ADD(RCYC, RCYC, W0);

            load_status_from_mem();

            instructionsInterpreted++;

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
   
   regman.pop_stashed_regs();
   emit.ABI_PopRegisters({30, 28, 19, 20, 21, 22, 23, 24, 25, 26, 27});
   emit.RET();

   void* fn_ptr = jitRXAddr + jitUsed;
   //printf("compiled block %d %p\n", jitUsed, fn_ptr);
   JIT_COMPILED_FUNC(base, PROCNUM) = (uintptr_t)fn_ptr;
   jitUsed = emit.GetCodePtr() - jitRWAddr;

   /*FILE* instruction_dump = fopen("instruction_dump", "w");
   fwrite(fn_ptr, (jitUsed - prevUsed), 1, instruction_dump);
   fclose(instruction_dump);*/
   
   jitTransitionToExecutable(&jitPage);
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

   printf("statistics: %.2f%% interpreted %.2f%% thumb\n", 
      (float)instructionsInterpreted/(float)instructionsTotal*100.f,
      (float)instructionsThumb/(float)instructionsTotal*100.f);
   instructionsTotal = 0;
   instructionsInterpreted = 0;
   instructionsThumb = 0;

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
      emit.SetCodePtrUnsafe(jitRWAddr);
   }
}

void arm_jit_close()
{
   if(jitUsed != -1)
      jitClose(&jitPage);
}
#endif // HAVE_JIT
