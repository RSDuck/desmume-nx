/*
	Copyright (C) 2006 yopyop
	Copyright (C) 2011 Loren Merritt
	Copyright (C) 2012-2017 DeSmuME team

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

#include "utils/bits.h"
#include "emitter/Arm64Emitter.h"
#include "armcpu.h"
#include "instructions.h"
#include "instruction_attributes.h"
#include "MMU.h"
#include "MMU_timing.h"
#include "arm_jit.h"
#include "bios.h"

#include "regman.h"

namespace libnx
{
#include <switch.h>
}

#define LOG_JIT_LEVEL 0
#define PROFILER_JIT_LEVEL 0

#if (PROFILER_JIT_LEVEL > 0)
#include <algorithm>
#endif

using namespace Arm64Gen;

#if (LOG_JIT_LEVEL > 0)
#define LOG_JIT 1
#define JIT_COMMENT(...) c.comment(__VA_ARGS__)
#define printJIT(buf, val) { \
	JIT_COMMENT("printJIT(\""##buf"\", val);"); \
	GpVar txt = c.newGpVar(kX86VarTypeGpz); \
	GpVar data = c.newGpVar(kX86VarTypeGpz); \
	GpVar io = c.newGpVar(kX86VarTypeGpd); \
	c.lea(io, dword_ptr_abs(stdout)); \
	c.lea(txt, dword_ptr_abs(&buf)); \
	c.mov(data, *(GpVar*)&val); \
	X86CompilerFuncCall* prn = c.call((uintptr_t)fprintf); \
	prn->setPrototype(kX86FuncConvDefault, FuncBuilder3<void, void*, void*, u32>()); \
	prn->setArgument(0, io); \
	prn->setArgument(1, txt); \
	prn->setArgument(2, data); \
	X86CompilerFuncCall* prn_flush = c.call((uintptr_t)fflush); \
	prn_flush->setPrototype(kX86FuncConvDefault, FuncBuilder1<void, void*>()); \
	prn_flush->setArgument(0, io); \
}
#else
#define LOG_JIT 0
#define JIT_COMMENT(...)
#define printJIT(buf, val)
#endif

u32 saveBlockSizeJIT = 0;

#ifdef MAPPED_JIT_FUNCS
CACHE_ALIGN JIT_struct JIT;

uintptr_t *JIT_struct::JIT_MEM[2][0x4000] = {{0}};

static uintptr_t *JIT_MEM[2][32] = {
	//arm9
	{
		/* 0X*/	DUP2(JIT.ARM9_ITCM),
		/* 1X*/	DUP2(JIT.ARM9_ITCM), // mirror
		/* 2X*/	DUP2(JIT.MAIN_MEM),
		/* 3X*/	DUP2(JIT.SWIRAM),
		/* 4X*/	DUP2(NULL),
		/* 5X*/	DUP2(NULL),
		/* 6X*/		 NULL, 
					 JIT.ARM9_LCDC,	// Plain ARM9-CPU Access (LCDC mode) (max 656KB)
		/* 7X*/	DUP2(NULL),
		/* 8X*/	DUP2(NULL),
		/* 9X*/	DUP2(NULL),
		/* AX*/	DUP2(NULL),
		/* BX*/	DUP2(NULL),
		/* CX*/	DUP2(NULL),
		/* DX*/	DUP2(NULL),
		/* EX*/	DUP2(NULL),
		/* FX*/	DUP2(JIT.ARM9_BIOS)
	},
	//arm7
	{
		/* 0X*/	DUP2(JIT.ARM7_BIOS),
		/* 1X*/	DUP2(NULL),
		/* 2X*/	DUP2(JIT.MAIN_MEM),
		/* 3X*/	     JIT.SWIRAM,
		             JIT.ARM7_ERAM,
		/* 4X*/	     NULL,
		             JIT.ARM7_WIRAM,
		/* 5X*/	DUP2(NULL),
		/* 6X*/		 JIT.ARM7_WRAM,		// VRAM allocated as Work RAM to ARM7 (max. 256K)
					 NULL,
		/* 7X*/	DUP2(NULL),
		/* 8X*/	DUP2(NULL),
		/* 9X*/	DUP2(NULL),
		/* AX*/	DUP2(NULL),
		/* BX*/	DUP2(NULL),
		/* CX*/	DUP2(NULL),
		/* DX*/	DUP2(NULL),
		/* EX*/	DUP2(NULL),
		/* FX*/	DUP2(NULL)
		}
};

static u32 JIT_MASK[2][32] = {
	//arm9
	{
		/* 0X*/	DUP2(0x00007FFF),
		/* 1X*/	DUP2(0x00007FFF),
		/* 2X*/	DUP2(0x003FFFFF), // FIXME _MMU_MAIN_MEM_MASK
		/* 3X*/	DUP2(0x00007FFF),
		/* 4X*/	DUP2(0x00000000),
		/* 5X*/	DUP2(0x00000000),
		/* 6X*/		 0x00000000,
					 0x000FFFFF,
		/* 7X*/	DUP2(0x00000000),
		/* 8X*/	DUP2(0x00000000),
		/* 9X*/	DUP2(0x00000000),
		/* AX*/	DUP2(0x00000000),
		/* BX*/	DUP2(0x00000000),
		/* CX*/	DUP2(0x00000000),
		/* DX*/	DUP2(0x00000000),
		/* EX*/	DUP2(0x00000000),
		/* FX*/	DUP2(0x00007FFF)
	},
	//arm7
	{
		/* 0X*/	DUP2(0x00003FFF),
		/* 1X*/	DUP2(0x00000000),
		/* 2X*/	DUP2(0x003FFFFF),
		/* 3X*/	     0x00007FFF,
		             0x0000FFFF,
		/* 4X*/	     0x00000000,
		             0x0000FFFF,
		/* 5X*/	DUP2(0x00000000),
		/* 6X*/		 0x0003FFFF,
					 0x00000000,
		/* 7X*/	DUP2(0x00000000),
		/* 8X*/	DUP2(0x00000000),
		/* 9X*/	DUP2(0x00000000),
		/* AX*/	DUP2(0x00000000),
		/* BX*/	DUP2(0x00000000),
		/* CX*/	DUP2(0x00000000),
		/* DX*/	DUP2(0x00000000),
		/* EX*/	DUP2(0x00000000),
		/* FX*/	DUP2(0x00000000)
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

#define DEBUG_DYNARC(x) if (debug_disable_##x) { return 0; }
// ARM
#define debug_disable_arithmetic 		0
#define debug_disable_arithmetic_r 		1
#define debug_disable_arithmetic_s 		1
#define debug_disable_tst_teq_cmp_cmn 	1
#define debug_disable_mov 				1
#define debug_disable_mul				1
#define debug_disable_mrs				1
#define debug_disable_ldr				1
#define debug_disable_str				1
#define debug_disable_ldrd_strd_pre		1
#define debug_disable_ldrd_strd_post	1
#define debug_disable_swp				1
#define debug_disable_ldm_stm			1
#define debug_disable_ldm_stm2			1
#define debug_disable_b					1
#define debug_disable_bx				1
#define debug_disable_clz				1
#define debug_disable_mrc				1
#define debug_disable_swi				1
#define debug_disable_thumb_shift_imm	1
#define debug_disable_thumb_shift_reg	1
#define debug_disable_thumb_shift_0		1
#define debug_disable_thumb_ror			1
#define debug_disable_thumb_logic		1
#define debug_disable_thumb_neg			1
#define debug_disable_thumb_add			1
#define debug_disable_thumb_sub			1
#define debug_disable_thumb_adc_sbc		1
#define debug_disable_thumb_mov			1
#define debug_disable_thumb_mul			1
#define debug_disable_thumb_cmp_tst		1
#define debug_disable_thumb_str			1
#define debug_disable_thumb_ldr			1
#define debug_disable_thumb_ldm_stm		1
#define debug_disable_thumb_adjust_sp	1
#define debug_disable_thumb_push_pop	1
#define debug_disable_thumb_branch		1
// Thumb

static u8 recompile_counts[(1<<26)/16];

static ARM64XEmitter c;

//static void emit_branch(int cond, Label to);
static bool emit_branch(int cond, FixupBranch& branch);
static void _armlog(u8 proc, u32 addr, u32 opcode);

//static FileLogger logger(stderr);

static int PROCNUM;
static int *PROCNUM_ptr = &PROCNUM;
static int bb_opcodesize;
static int bb_adr;
static bool bb_thumb;
static u32 bb_constant_cycles;

static u32* jit_rw_addr = nullptr;
static u32* jit_rx_addr = nullptr;
static libnx::Jit jit_page;

static u8* branched_code_buf = nullptr;

static const int Jit_Size = 0x400000; // 4 Mb

// for register usage see regman.h
static reg_manager regman;

#define cpu (&ARMPROC)
#define bb_next_instruction (bb_adr + bb_opcodesize)
#define bb_r15				(bb_adr + 2 * bb_opcodesize)

#define map_reg(x)			regman.map_reg32(REG_POS(i, (x)))
#define map_reg_thumb(x)	regman.map_reg32((i>>(x))&0x7)

#define mov_reg_L(rd, x)	c.SBFM(rd, map_reg(x), 0, 15)
#define mov_reg_H(rd, x)	c.SBFM(rd, map_reg(x), 16, 31)

#define cpu_ptr(x)			dword_ptr(bb_cpu, offsetof(armcpu_t, x))
#define cpu_ptr_byte(x, y)	byte_ptr(bb_cpu, offsetof(armcpu_t, x) + y)
#define flags_ptr			cpu_ptr_byte(CPSR.val, 3)
#define reg_ptr(x)			dword_ptr(bb_cpu, offsetof(armcpu_t, R) + 4*(x))
#define reg_pos_ptr(x)		dword_ptr(bb_cpu, offsetof(armcpu_t, R) + 4*REG_POS(i,(x)))
#define reg_pos_ptrL(x)		word_ptr( bb_cpu, offsetof(armcpu_t, R) + 4*REG_POS(i,(x)))
#define reg_pos_ptrH(x)		word_ptr( bb_cpu, offsetof(armcpu_t, R) + 4*REG_POS(i,(x)) + 2)
#define reg_pos_ptrB(x)		byte_ptr( bb_cpu, offsetof(armcpu_t, R) + 4*REG_POS(i,(x)))
#define reg_pos_thumb(x)	dword_ptr(bb_cpu, offsetof(armcpu_t, R) + 4*((i>>(x))&0x7))
#define reg_pos_thumbB(x)	byte_ptr(bb_cpu, offsetof(armcpu_t, R) + 4*((i>>(x))&0x7))
#define cp15_ptr(x)			dword_ptr(bb_cp15, offsetof(armcp15_t, x))
#define cp15_ptr_off(x, y)	dword_ptr(bb_cp15, offsetof(armcp15_t, x) + y)
#define mmu_ptr(x)			dword_ptr(bb_mmu, offsetof(MMU_struct, x))
#define mmu_ptr_byte(x)		byte_ptr(bb_mmu, offsetof(MMU_struct, x))
#define _REG_NUM(i, n)		((i>>(n))&0x7)

#ifndef ASMJIT_X64
#define r64 r32
#endif

// sequencer.reschedule = true;
#define changeCPSR { \
			regman.call(X2, NDS_Reschedule, false); \
}

#if (PROFILER_JIT_LEVEL > 0)
struct PROFILER_COUNTER_INFO
{
	u64	count;
	char name[64];
};

struct JIT_PROFILER
{
	JIT_PROFILER()
	{
		memset(&arm_count[0], 0, sizeof(arm_count));
		memset(&thumb_count[0], 0, sizeof(thumb_count));
	}

	u64 arm_count[4096];
	u64 thumb_count[1024];
} profiler_counter[2];

static GpVar bb_profiler;

#define profiler_counter_arm(opcode)   qword_ptr(bb_profiler, offsetof(JIT_PROFILER, arm_count) + (INSTRUCTION_INDEX(opcode)*sizeof(u64)))
#define profiler_counter_thumb(opcode) qword_ptr(bb_profiler, offsetof(JIT_PROFILER, thumb_count) + ((opcode>>6)*sizeof(u64)))

#if (PROFILER_JIT_LEVEL > 1)
struct PROFILER_ENTRY
{
	u32 addr;
	u32	cycles;
} profiler_entry[2][1<<26];

static GpVar bb_profiler_entry;
#endif

#endif
//-----------------------------------------------------------------------------
//   Shifting macros
//-----------------------------------------------------------------------------
#define SET_NZCV { \
	JIT_COMMENT("SET_NZCV"); \
	auto x = regman.alloc_temp32(); \
	c.MRS(EncodeRegTo64(x), FIELD_NZCV); \
	c.BFI(RCPSR, x, 28, 4); \
	regman.free_temp32(x); \
	regman.mark_cpsr_dirty(); \
	JIT_COMMENT("end SET_NZCV"); \
}

#define SET_NZC { \
	JIT_COMMENT("SET_NZC"); \
	auto tmp = regman.alloc_temp32(); \
	c.CSET(tmp, CC_NEQ); \
	c.BFI(RCPSR, tmp, 30, 1); \
	c.CSET(tmp, CC_PL); \
	c.BFI(RCPSR, tmp, 31, 1); \
	regman.free_temp32(tmp); \
	if (cf_change) { c.BFI(RCPSR, rcf, 29, 1); regman.free_temp32(rcf); } \
	regman.mark_cpsr_dirty(); \
	JIT_COMMENT("end SET_NZC"); \
}

#define SET_NZ(clear_cv) { \
	JIT_COMMENT("SET_NZ"); \
	auto tmp = regman.alloc_temp32(); \
	c.CSET(tmp, CC_NEQ); \
	c.BFI(RCPSR, tmp, 30, 1); \
	c.CSET(tmp, CC_PL); \
	c.BFI(RCPSR, tmp, 31, 1); \
	if (clear_cv) { c.ANDI2R(RCPSR, RCPSR, 0xCFFFFFFF, tmp); } \
	regman.free_temp32(tmp); \
	regman.mark_cpsr_dirty(); \
	JIT_COMMENT("end SET_NZ"); \
}

#define SET_N { \
	JIT_COMMENT("SET_N"); \
	GpVar x = c.newGpVar(kX86VarTypeGpz); \
	GpVar y = c.newGpVar(kX86VarTypeGpz); \
	c.sets(x.r8Lo()); \
	c.movzx(y, flags_ptr); \
	c.and_(y, 0x7F); \
	c.shl(x, 7); \
	c.or_(x, y); \
	c.mov(flags_ptr, x.r8Lo()); \
	JIT_COMMENT("end SET_N"); \
}

#define SET_Z { \
	JIT_COMMENT("SET_Z"); \
	GpVar x = c.newGpVar(kX86VarTypeGpz); \
	GpVar y = c.newGpVar(kX86VarTypeGpz); \
	c.setz(x.r8Lo()); \
	c.movzx(y, flags_ptr); \
	c.and_(y, 0xBF); \
	c.shl(x, 6); \
	c.or_(x, y); \
	c.mov(flags_ptr, x.r8Lo()); \
	JIT_COMMENT("end SET_Z"); \
}

#define SET_Q { \
	JIT_COMMENT("SET_Q"); \
	auto x = regman.alloc_temp32(); \
	c.CSET(x, CC_CC); \
	c.BFI(RCPSR, x, 29, 1); \
	regman.free_temp32(x); \
	regman.mark_cpsr_dirty(); \
	JIT_COMMENT("end SET_Q"); \
}

#define S_DST_R15 { \
	JIT_COMMENT("S_DST_R15"); \
	auto tmp = regman.alloc_temp32(); \
	c.LDR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, SPSR.val)); \
	c.MOV(X0, RCPU); \
	c.ANDI2R(X1, tmp, 0x1f); \
	regman.call(X2, armcpu_switchMode, true); \
	c.MOV(RCPSR, tmp); \
	regman.mark_cpsr_dirty(); \
	/* next_instruction = r15 & (0xFFFFFFFC | CPSR.thumb << 1)*/ \
	c.UBFX(tmp, tmp, 5, 1); \
	c.BFI(tmp, tmp, 1, 1); \
	c.ORRI2R(tmp, tmp, 0xFFFFFFFC); \
	c.AND(tmp, regman.map_reg32(15), tmp); \
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction)); \
	regman.free_temp32(tmp); \
}

// ============================================================================================= IMM
#define LSL_IMM \
	JIT_COMMENT("LSL_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u32 imm = ((i>>7)&0x1F); \
	auto rhs = regman.alloc_temp32(); \
	c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_LSL, imm)); \
	u32 rhs_first = cpu->R[REG_POS(i,0)] << imm;

#define S_LSL_IMM \
	JIT_COMMENT("S_LSL_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u8 cf_change = 0; \
	u32 imm = ((i>>7)&0x1F); \
	auto rhs = regman.alloc_temp32(); \
	c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_LSL, imm)); \
	ARM64Reg rcf; \
	if (imm)  \
	{ \
		cf_change = 1; \
		rcf = regman.alloc_temp32(); \
		c.UBFX(rcf, map_reg(0), 32 - imm, 1); \
	}

#define LSR_IMM \
	JIT_COMMENT("LSR_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u32 imm = ((i>>7)&0x1F); \
	auto rhs = regman.alloc_temp32(); \
	if(imm) \
		c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_LSR, imm)); \
	else \
		c.MOVZ(rhs, 0); \
	u32 rhs_first = imm ? cpu->R[REG_POS(i,0)] >> imm : 0;

#define S_LSR_IMM \
	JIT_COMMENT("S_LSR_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u8 cf_change = 1; \
	auto rcf = regman.alloc_temp32(); \
	auto rhs = regman.alloc_temp32(); \
	u32 imm = ((i>>7)&0x1F); \
	if (!imm) \
	{ \
		c.MOVZ(rhs, 0); \
		c.UBFX(rcf, map_reg(0), 31, 1); \
	} \
	else \
	{ \
		c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_LSR, imm)); \
		c.UBFX(rcf, map_reg(0), imm-1, 1); \
	}

#define ASR_IMM \
	JIT_COMMENT("ASR_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u32 imm = ((i>>7)&0x1F); \
	auto rhs = regman.alloc_temp32(); \
	c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_ASR, imm?imm:31)); \
	u32 rhs_first = (s32)cpu->R[REG_POS(i,0)] >> imm;

#define S_ASR_IMM \
	JIT_COMMENT("S_ASR_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u8 cf_change = 1; \
	auto rcf = regman.alloc_temp32(); \
	auto rhs = regman.alloc_temp32(); \
	u32 imm = ((i>>7)&0x1F); \
	c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_ASR, imm?imm:31)); \
	c.UBFX(rcf, map_reg(0), imm?imm-1:31, 1);

#define ROR_IMM \
	JIT_COMMENT("ROR_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u32 imm = ((i>>7)&0x1F); \
	auto rhs = regman.alloc_temp32(); \
	c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_ROR, !imm ? 1 : imm)); \
	if (!imm) \
	{ \
		auto cf = regman.alloc_temp32(); \
		c.UBFX(cf, RCPSR, 29, 1); \
		c.BFI(rhs, cf, 31, 1); \
		regman.free_temp32(cf); \
	} \
	u32 rhs_first = imm?ROR(cpu->R[REG_POS(i,0)], imm) : ((u32)cpu->CPSR.bits.C<<31)|(cpu->R[REG_POS(i,0)]>>1);

#define S_ROR_IMM \
	JIT_COMMENT("S_ROR_IMM"); \
	regman.load_reg(REG_POS(i, 0)); \
	bool rhs_is_imm = false; \
	u8 cf_change = 1; \
	auto rhs = regman.alloc_temp32(); \
	auto rcf = regman.alloc_temp32(); \
	u32 imm = ((i>>7)&0x1F); \
	c.MOV(rhs, map_reg(0), ArithOption(rhs, ST_ROR, !imm ? 1 : imm)); \
	if (!imm) \
	{ \
		c.UBFX(rcf, RCPSR, 29, 1); \
		c.BFI(rhs, rcf, 31, 1); \
	} \
	c.UBFX(rcf, map_reg(0), !imm?0:(imm-1), 1);

#define REG_OFF \
	JIT_COMMENT("REG_OFF"); \
	bool rhs_is_imm = false; \
	auto rhs = regman.alloc_temp32(); \
	c.MOV(rhs, map_reg(0)); \
	u32 rhs_first = cpu->R[REG_POS(i,0)];

#define IMM_VAL \
	JIT_COMMENT("IMM_VAL"); \
	bool rhs_is_imm = true; \
	u32 imm_eval = ROR((i&0xFF), (i>>7)&0x1E); \
	auto rhs = regman.alloc_temp32(); \
	c.MOVI2R(rhs, imm_eval); \
	u32 rhs_first = imm_eval;

#define S_IMM_VAL \
	JIT_COMMENT("S_IMM_VAL"); \
	bool rhs_is_imm = true; \
	u8 cf_change = 0; \
	ARM64Reg rcf; \
	u32 imm_eval = ROR((i&0xFF), (i>>7)&0x1E); \
	auto rhs = regman.alloc_temp32(); \
	c.MOVI2R(rhs, imm_eval); \
	if ((i>>8)&0xF) \
	{ \
		cf_change = 1; \
		rcf = regman.alloc_temp32(); \
		c.MOVZ(rcf, BIT31(rhs)); \
	} \
	u32 rhs_first = imm_eval;

#define IMM_OFF \
	JIT_COMMENT("IMM_OFF"); \
	bool rhs_is_imm = true; \
	u32 imm = ((i>>4)&0xF0)+(i&0xF); \
	u32 rhs_first = imm; \
	auto rhs = regman.alloc_temp32(); \
	c.MOVI2R(rhs, imm);

#define IMM_OFF_12 \
	JIT_COMMENT("IMM_OFF_12"); \
	bool rhs_is_imm = true; \
	u32 imm = (i & 0xFFF); \
	auto rhs = regman.alloc_temp32(); \
	c.MOVI2R(rhs, imm); \
	u32 rhs_first = imm;

// ============================================================================================= REG
#define LSX_REG(name, a64inst, sign) \
	JIT_COMMENT(#name); \
	regman.load_reg(REG_POS(i, 0)); \
	regman.load_reg(REG_POS(i, 8)); \
	bool rhs_is_imm = false; \
	auto rhs = regman.alloc_temp32(); \
	auto tmp = regman.alloc_temp32(); \
	c.ANDI2R(tmp, map_reg(8), 0xff); \
	c.a64inst(rhs, map_reg(0), tmp); \
	c.CMP(tmp, 32); \
	if (!sign) \
	{ \
		c.MOVZ(tmp, 0); \
		c.CSEL(rhs, rhs, tmp, CC_LT); \
	} \
	else \
	{ \
		c.MOV(tmp, map_reg(0), ArithOption(tmp, ST_ASR, 31)); \
		c.CSEL(rhs, rhs, tmp, CC_LT); \
	} \
	regman.free_temp32(tmp);

#define S_LSX_REG(name, a64inst, sign, lshift) \
	JIT_COMMENT(#name); \
	regman.load_reg(REG_POS(i, 0)); \
	regman.load_reg(REG_POS(i, 8)); \
	bool rhs_is_imm = false; \
	u8 cf_change = 1; \
	auto rhs = regman.alloc_temp32(); \
	auto rcf = regman.alloc_temp32(); \
	auto tmp = regman.alloc_temp32(); \
	c.UBFX(rcf, RCPSR, 29, 1); \
	c.UBFX(tmp, map_reg(8), 0, 8); \
	c.CMP(tmp, 32); \
	auto __skip_shift = c.CBZ(tmp); \
	c.SUB(tmp, tmp, 1); \
	c.a64inst##V(rhs, map_reg(0), tmp); \
	c.UBFX(rcf, rhs, lshift ? 31 : 0, 1); \
	c.a64inst(rhs, rhs, 1); \
	if (!sign) \
	{ \
		c.MOVZ(tmp, 0); \
		c.CSEL(rhs, rhs, tmp, CC_LT); \
	} \
	else \
	{ \
		c.MOV(tmp, map_reg(0), ArithOption(tmp, ST_ASR, 31)); \
		c.CSEL(rhs, rhs, tmp, CC_LT); \
	} \
	c.SetJumpTarget(__skip_shift); \
	regman.free_temp32(tmp);

#define LSL_REG LSX_REG(LSL_REG, LSL, 0)
#define LSR_REG LSX_REG(LSR_REG, LSR, 0)
#define ASR_REG LSX_REG(ASR_REG, ASR, 1)
#define S_LSL_REG S_LSX_REG(S_LSL_REG, LSL, 0, 1)
#define S_LSR_REG S_LSX_REG(S_LSR_REG, LSR, 0, 0)
#define S_ASR_REG S_LSX_REG(S_ASR_REG, ASR, 1, 0)

#define ROR_REG \
	JIT_COMMENT("ROR_REG"); \
	regman.load_reg(REG_POS(i, 0)); \
	regman.load_reg(REG_POS(i, 8)); \
	bool rhs_is_imm = false; \
	auto rhs = regman.alloc_temp32(); \
	auto tmp = regman.alloc_temp32(); \
	c.ANDI2R(tmp, map_reg(8), 0xff); \
	c.RORV(rhs, map_reg(0), tmp); \
	regman.free_temp32(tmp);

#define S_ROR_REG \
	JIT_COMMENT("S_ROR_REG"); \
	regman.load_reg(REG_POS(i, 0)); \
	regman.load_reg(REG_POS(i, 8)); \
	bool rhs_is_imm = false; \
	bool cf_change = 1; \
	auto rhs = regman.alloc_temp32(); \
	auto tmp = regman.alloc_temp32(); \
	auto rcf = regman.alloc_temp32(); \
	c.UBFX(rcf, RCPSR, 29, 1); \
	c.UBFX(tmp, map_reg(8), 0, 8); \
	auto __zero = c.CBZ(tmp); \
	c.SUB(tmp, tmp, 1); \
	c.RORV(rhs, map_reg(0), tmp); \
	c.UBFX(rcf, rhs, 0, 1); \
	c.ROR(rhs, rhs, 1); \
	c.SetJumpTarget(__zero); \
	regman.free_temp32(tmp);

//==================================================================== common funcs
static void emit_MMU_aluMemCycles(int alu_cycles, ARM64Reg mem_cycles, int population)
{
	if(PROCNUM==ARMCPU_ARM9)
	{
		if(population < alu_cycles)
		{
			// mem_cycles = max(alu_cycles, mem_cycles);
			auto x = regman.alloc_temp32();
			c.MOVI2R(x, alu_cycles);
			c.CMP(mem_cycles, x);
			c.CSEL(mem_cycles, x, mem_cycles, CC_LT);
			regman.free_temp32(x);
		}
	}
	else
		c.ADD(mem_cycles, mem_cycles, alu_cycles);
}

//-----------------------------------------------------------------------------
//   OPs
//-----------------------------------------------------------------------------
#define OP_ARITHMETIC(arg, a64inst, symmetric, flags) \
	DEBUG_DYNARC(arithmetic) \
	if(flags) return 0;\
    arg; \
	regman.load_reg(REG_POS(i,12), true); \
	regman.load_reg(REG_POS(i,16)); \
	c.a64inst(map_reg(12), map_reg(16), rhs); \
	regman.free_temp32(rhs); \
	if(flags) \
	{ \
		if(REG_POS(i,12)==15) \
		{ \
			S_DST_R15; \
			c.ADD(Rtotal_cycles, Rtotal_cycles, 2); \
			return 1; \
		} \
		SET_NZCV \
	} \
	else \
	{ \
		if(REG_POS(i,12)==15) \
		{ \
			c.STR(INDEX_UNSIGNED, regman.map_reg32(15), RCPU, offsetof(armcpu_t, next_instruction)); \
			c.ADD(Rtotal_cycles, Rtotal_cycles, 2); \
		} \
	} \
	return 1;

#define OP_ARITHMETIC_R(arg, a64inst, flags) \
	DEBUG_DYNARC(arithmetic_r) \
    arg; \
	c.a64inst(map_reg(12), rhs, map_reg(16)); \
	regman.free_temp32(rhs); \
	if(flags) \
	{ \
		if(REG_POS(i,12)==15) \
		{ \
			S_DST_R15; \
			c.ADD(Rtotal_cycles, Rtotal_cycles, 2); \
			return 1; \
		} \
		SET_NZCV \
	} \
	else \
	{ \
		if(REG_POS(i,12)==15) \
		{ \
			c.STR(INDEX_UNSIGNED, map_reg(15), RCPU, offsetof(armcpu_t, next_instruction)); \
			c.ADD(Rtotal_cycles, Rtotal_cycles, 2); \
		} \
	} \
	return 1;

#define OP_ARITHMETIC_S(arg, a64inst, symmetric) \
	DEBUG_DYNARC(arithmetic_s) \
    arg; \
	c.a64inst(map_reg(12), map_reg(16), rhs); \
	if(REG_POS(i,12)==15) \
	{ \
		S_DST_R15; \
		c.ADD(Rtotal_cycles, Rtotal_cycles, 2); \
		return 1; \
	} \
	c.TST(map_reg(12), map_reg(12)); \
	SET_NZC; \
	return 1;

#define GET_CARRY { \
	c._MSR(FIELD_NZCV, EncodeRegTo64(RCPSR)); }

static int OP_AND_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, AND, 1, 0); }
static int OP_AND_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, AND, 1, 0); }
static int OP_AND_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, AND, 1, 0); }
static int OP_AND_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, AND, 1, 0); }
static int OP_AND_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, AND, 1, 0); }
static int OP_AND_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, AND, 1, 0); }
static int OP_AND_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, AND, 1, 0); }
static int OP_AND_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, AND, 1, 0); }
static int OP_AND_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, AND, 1, 0); }

static int OP_EOR_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, EOR, 1, 0); }
static int OP_EOR_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, EOR, 1, 0); }
static int OP_EOR_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, EOR, 1, 0); }
static int OP_EOR_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, EOR, 1, 0); }
static int OP_EOR_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, EOR, 1, 0); }
static int OP_EOR_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, EOR, 1, 0); }
static int OP_EOR_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, EOR, 1, 0); }
static int OP_EOR_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, EOR, 1, 0); }
static int OP_EOR_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, EOR, 1, 0); }

static int OP_ORR_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, ORR, 1, 0); }
static int OP_ORR_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, ORR, 1, 0); }
static int OP_ORR_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, ORR, 1, 0); }
static int OP_ORR_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, ORR, 1, 0); }
static int OP_ORR_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, ORR, 1, 0); }
static int OP_ORR_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, ORR, 1, 0); }
static int OP_ORR_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, ORR, 1, 0); }
static int OP_ORR_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, ORR, 1, 0); }
static int OP_ORR_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, ORR, 1, 0); }

static int OP_ADD_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, ADD, 1, 0); }
static int OP_ADD_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, ADD, 1, 0); }
static int OP_ADD_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, ADD, 1, 0); }
static int OP_ADD_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, ADD, 1, 0); }
static int OP_ADD_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, ADD, 1, 0); }
static int OP_ADD_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, ADD, 1, 0); }
static int OP_ADD_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, ADD, 1, 0); }
static int OP_ADD_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, ADD, 1, 0); }
static int OP_ADD_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, ADD, 1, 0); }

static int OP_SUB_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, SUB, 0, 0); }
static int OP_SUB_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, SUB, 0, 0); }
static int OP_SUB_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, SUB, 0, 0); }
static int OP_SUB_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, SUB, 0, 0); }
static int OP_SUB_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, SUB, 0, 0); }
static int OP_SUB_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, SUB, 0, 0); }
static int OP_SUB_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, SUB, 0, 0); }
static int OP_SUB_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, SUB, 0, 0); }
static int OP_SUB_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, SUB, 0, 0); }

static int OP_RSB_LSL_IMM(const u32 i) { OP_ARITHMETIC_R(LSL_IMM, SUB, 0); }
static int OP_RSB_LSL_REG(const u32 i) { OP_ARITHMETIC_R(LSL_REG, SUB, 0); }
static int OP_RSB_LSR_IMM(const u32 i) { OP_ARITHMETIC_R(LSR_IMM, SUB, 0); }
static int OP_RSB_LSR_REG(const u32 i) { OP_ARITHMETIC_R(LSR_REG, SUB, 0); }
static int OP_RSB_ASR_IMM(const u32 i) { OP_ARITHMETIC_R(ASR_IMM, SUB, 0); }
static int OP_RSB_ASR_REG(const u32 i) { OP_ARITHMETIC_R(ASR_REG, SUB, 0); }
static int OP_RSB_ROR_IMM(const u32 i) { OP_ARITHMETIC_R(ROR_IMM, SUB, 0); }
static int OP_RSB_ROR_REG(const u32 i) { OP_ARITHMETIC_R(ROR_REG, SUB, 0); }
static int OP_RSB_IMM_VAL(const u32 i) { OP_ARITHMETIC_R(IMM_VAL, SUB, 0); }

// ================================ S instructions
static int OP_AND_S_LSL_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSL_IMM, AND, 1); }
static int OP_AND_S_LSL_REG(const u32 i) { OP_ARITHMETIC_S(S_LSL_REG, AND, 1); }
static int OP_AND_S_LSR_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSR_IMM, AND, 1); }
static int OP_AND_S_LSR_REG(const u32 i) { OP_ARITHMETIC_S(S_LSR_REG, AND, 1); }
static int OP_AND_S_ASR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ASR_IMM, AND, 1); }
static int OP_AND_S_ASR_REG(const u32 i) { OP_ARITHMETIC_S(S_ASR_REG, AND, 1); }
static int OP_AND_S_ROR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ROR_IMM, AND, 1); }
static int OP_AND_S_ROR_REG(const u32 i) { OP_ARITHMETIC_S(S_ROR_REG, AND, 1); }
static int OP_AND_S_IMM_VAL(const u32 i) { OP_ARITHMETIC_S(S_IMM_VAL, AND, 1); }

static int OP_EOR_S_LSL_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSL_IMM, EOR, 1); }
static int OP_EOR_S_LSL_REG(const u32 i) { OP_ARITHMETIC_S(S_LSL_REG, EOR, 1); }
static int OP_EOR_S_LSR_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSR_IMM, EOR, 1); }
static int OP_EOR_S_LSR_REG(const u32 i) { OP_ARITHMETIC_S(S_LSR_REG, EOR, 1); }
static int OP_EOR_S_ASR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ASR_IMM, EOR, 1); }
static int OP_EOR_S_ASR_REG(const u32 i) { OP_ARITHMETIC_S(S_ASR_REG, EOR, 1); }
static int OP_EOR_S_ROR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ROR_IMM, EOR, 1); }
static int OP_EOR_S_ROR_REG(const u32 i) { OP_ARITHMETIC_S(S_ROR_REG, EOR, 1); }
static int OP_EOR_S_IMM_VAL(const u32 i) { OP_ARITHMETIC_S(S_IMM_VAL, EOR, 1); }

static int OP_ORR_S_LSL_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSL_IMM, ORR, 1); }
static int OP_ORR_S_LSL_REG(const u32 i) { OP_ARITHMETIC_S(S_LSL_REG, ORR, 1); }
static int OP_ORR_S_LSR_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSR_IMM, ORR, 1); }
static int OP_ORR_S_LSR_REG(const u32 i) { OP_ARITHMETIC_S(S_LSR_REG, ORR, 1); }
static int OP_ORR_S_ASR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ASR_IMM, ORR, 1); }
static int OP_ORR_S_ASR_REG(const u32 i) { OP_ARITHMETIC_S(S_ASR_REG, ORR, 1); }
static int OP_ORR_S_ROR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ROR_IMM, ORR, 1); }
static int OP_ORR_S_ROR_REG(const u32 i) { OP_ARITHMETIC_S(S_ROR_REG, ORR, 1); }
static int OP_ORR_S_IMM_VAL(const u32 i) { OP_ARITHMETIC_S(S_IMM_VAL, ORR, 1); }

static int OP_ADD_S_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, ADDS, 1, 1); }
static int OP_ADD_S_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, ADDS, 1, 1); }
static int OP_ADD_S_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, ADDS, 1, 1); }
static int OP_ADD_S_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, ADDS, 1, 1); }
static int OP_ADD_S_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, ADDS, 1, 1); }
static int OP_ADD_S_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, ADDS, 1, 1); }
static int OP_ADD_S_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, ADDS, 1, 1); }
static int OP_ADD_S_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, ADDS, 1, 1); }
static int OP_ADD_S_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, ADDS, 1, 1); }

static int OP_SUB_S_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, SUBS, 0, 1); }
static int OP_SUB_S_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, SUBS, 0, 1); }
static int OP_SUB_S_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, SUBS, 0, 1); }
static int OP_SUB_S_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, SUBS, 0, 1); }
static int OP_SUB_S_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, SUBS, 0, 1); }
static int OP_SUB_S_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, SUBS, 0, 1); }
static int OP_SUB_S_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, SUBS, 0, 1); }
static int OP_SUB_S_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, SUBS, 0, 1); }
static int OP_SUB_S_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, SUBS, 0, 1); }

static int OP_RSB_S_LSL_IMM(const u32 i) { OP_ARITHMETIC_R(LSL_IMM, SUBS, 1); }
static int OP_RSB_S_LSL_REG(const u32 i) { OP_ARITHMETIC_R(LSL_REG, SUBS, 1); }
static int OP_RSB_S_LSR_IMM(const u32 i) { OP_ARITHMETIC_R(LSR_IMM, SUBS, 1); }
static int OP_RSB_S_LSR_REG(const u32 i) { OP_ARITHMETIC_R(LSR_REG, SUBS, 1); }
static int OP_RSB_S_ASR_IMM(const u32 i) { OP_ARITHMETIC_R(ASR_IMM, SUBS, 1); }
static int OP_RSB_S_ASR_REG(const u32 i) { OP_ARITHMETIC_R(ASR_REG, SUBS, 1); }
static int OP_RSB_S_ROR_IMM(const u32 i) { OP_ARITHMETIC_R(ROR_IMM, SUBS, 1); }
static int OP_RSB_S_ROR_REG(const u32 i) { OP_ARITHMETIC_R(ROR_REG, SUBS, 1); }
static int OP_RSB_S_IMM_VAL(const u32 i) { OP_ARITHMETIC_R(IMM_VAL, SUBS, 1); }

static int OP_ADC_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG; GET_CARRY, ADC, 1, 0); }
static int OP_ADC_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL; GET_CARRY, ADC, 1, 0); }

static int OP_ADC_S_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG; GET_CARRY, ADCS, 1, 1); }
static int OP_ADC_S_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL; GET_CARRY, ADCS, 1, 1); }

static int OP_SBC_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG; GET_CARRY, SBC, 0, 0); }
static int OP_SBC_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL; GET_CARRY, SBC, 0, 0); }

static int OP_SBC_S_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG; GET_CARRY, SBC, 0, 1); }
static int OP_SBC_S_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL; GET_CARRY, SBC, 0, 1); }

static int OP_RSC_LSL_IMM(const u32 i) { OP_ARITHMETIC_R(LSL_IMM; GET_CARRY, SBC, 0); }
static int OP_RSC_LSL_REG(const u32 i) { OP_ARITHMETIC_R(LSL_REG; GET_CARRY, SBC, 0); }
static int OP_RSC_LSR_IMM(const u32 i) { OP_ARITHMETIC_R(LSR_IMM; GET_CARRY, SBC, 0); }
static int OP_RSC_LSR_REG(const u32 i) { OP_ARITHMETIC_R(LSR_REG; GET_CARRY, SBC, 0); }
static int OP_RSC_ASR_IMM(const u32 i) { OP_ARITHMETIC_R(ASR_IMM; GET_CARRY, SBC, 0); }
static int OP_RSC_ASR_REG(const u32 i) { OP_ARITHMETIC_R(ASR_REG; GET_CARRY, SBC, 0); }
static int OP_RSC_ROR_IMM(const u32 i) { OP_ARITHMETIC_R(ROR_IMM; GET_CARRY, SBC, 0); }
static int OP_RSC_ROR_REG(const u32 i) { OP_ARITHMETIC_R(ROR_REG; GET_CARRY, SBC, 0); }
static int OP_RSC_IMM_VAL(const u32 i) { OP_ARITHMETIC_R(IMM_VAL; GET_CARRY, SBC, 0); }

static int OP_RSC_S_LSL_IMM(const u32 i) { OP_ARITHMETIC_R(LSL_IMM; GET_CARRY, SBC, 1); }
static int OP_RSC_S_LSL_REG(const u32 i) { OP_ARITHMETIC_R(LSL_REG; GET_CARRY, SBC, 1); }
static int OP_RSC_S_LSR_IMM(const u32 i) { OP_ARITHMETIC_R(LSR_IMM; GET_CARRY, SBC, 1); }
static int OP_RSC_S_LSR_REG(const u32 i) { OP_ARITHMETIC_R(LSR_REG; GET_CARRY, SBC, 1); }
static int OP_RSC_S_ASR_IMM(const u32 i) { OP_ARITHMETIC_R(ASR_IMM; GET_CARRY, SBC, 1); }
static int OP_RSC_S_ASR_REG(const u32 i) { OP_ARITHMETIC_R(ASR_REG; GET_CARRY, SBC, 1); }
static int OP_RSC_S_ROR_IMM(const u32 i) { OP_ARITHMETIC_R(ROR_IMM; GET_CARRY, SBC, 1); }
static int OP_RSC_S_ROR_REG(const u32 i) { OP_ARITHMETIC_R(ROR_REG; GET_CARRY, SBC, 1); }
static int OP_RSC_S_IMM_VAL(const u32 i) { OP_ARITHMETIC_R(IMM_VAL; GET_CARRY, SBC, 1); }

static int OP_BIC_LSL_IMM(const u32 i) { OP_ARITHMETIC(LSL_IMM, BIC, 1, 0); }
static int OP_BIC_LSL_REG(const u32 i) { OP_ARITHMETIC(LSL_REG, BIC, 1, 0); }
static int OP_BIC_LSR_IMM(const u32 i) { OP_ARITHMETIC(LSR_IMM, BIC, 1, 0); }
static int OP_BIC_LSR_REG(const u32 i) { OP_ARITHMETIC(LSR_REG, BIC, 1, 0); }
static int OP_BIC_ASR_IMM(const u32 i) { OP_ARITHMETIC(ASR_IMM, BIC, 1, 0); }
static int OP_BIC_ASR_REG(const u32 i) { OP_ARITHMETIC(ASR_REG, BIC, 1, 0); }
static int OP_BIC_ROR_IMM(const u32 i) { OP_ARITHMETIC(ROR_IMM, BIC, 1, 0); }
static int OP_BIC_ROR_REG(const u32 i) { OP_ARITHMETIC(ROR_REG, BIC, 1, 0); }
static int OP_BIC_IMM_VAL(const u32 i) { OP_ARITHMETIC(IMM_VAL, BIC, 1, 0); }

static int OP_BIC_S_LSL_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSL_IMM, AND, 1); }
static int OP_BIC_S_LSL_REG(const u32 i) { OP_ARITHMETIC_S(S_LSL_REG, AND, 1); }
static int OP_BIC_S_LSR_IMM(const u32 i) { OP_ARITHMETIC_S(S_LSR_IMM, AND, 1); }
static int OP_BIC_S_LSR_REG(const u32 i) { OP_ARITHMETIC_S(S_LSR_REG, AND, 1); }
static int OP_BIC_S_ASR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ASR_IMM, AND, 1); }
static int OP_BIC_S_ASR_REG(const u32 i) { OP_ARITHMETIC_S(S_ASR_REG, AND, 1); }
static int OP_BIC_S_ROR_IMM(const u32 i) { OP_ARITHMETIC_S(S_ROR_IMM, AND, 1); }
static int OP_BIC_S_ROR_REG(const u32 i) { OP_ARITHMETIC_S(S_ROR_REG, AND, 1); }
static int OP_BIC_S_IMM_VAL(const u32 i) { OP_ARITHMETIC_S(S_IMM_VAL, AND, 1); }

//-----------------------------------------------------------------------------
//   TST
//-----------------------------------------------------------------------------
#define OP_TST_(arg) \
	DEBUG_DYNARC(tst_teq_cmp_cmn) \
	arg; \
	c.TST(map_reg(16), rhs); \
	regman.free_temp32(rhs); \
	SET_NZC; \
	return 1;

static int OP_TST_LSL_IMM(const u32 i) { OP_TST_(S_LSL_IMM); }
static int OP_TST_LSL_REG(const u32 i) { OP_TST_(S_LSL_REG); }
static int OP_TST_LSR_IMM(const u32 i) { OP_TST_(S_LSR_IMM); }
static int OP_TST_LSR_REG(const u32 i) { OP_TST_(S_LSR_REG); }
static int OP_TST_ASR_IMM(const u32 i) { OP_TST_(S_ASR_IMM); }
static int OP_TST_ASR_REG(const u32 i) { OP_TST_(S_ASR_REG); }
static int OP_TST_ROR_IMM(const u32 i) { OP_TST_(S_ROR_IMM); }
static int OP_TST_ROR_REG(const u32 i) { OP_TST_(S_ROR_REG); }
static int OP_TST_IMM_VAL(const u32 i) { OP_TST_(S_IMM_VAL); }

//-----------------------------------------------------------------------------
//   TEQ
//-----------------------------------------------------------------------------
#define OP_TEQ_(arg) \
	DEBUG_DYNARC(tst_teq_cmp_cmn) \
	arg; \
	c.EOR(rhs, map_reg(16), rhs); /* recycle rhs */ \
	c.TST(rhs, rhs); \
	regman.free_temp32(rhs); \
	SET_NZC; \
	return 1;

static int OP_TEQ_LSL_IMM(const u32 i) { OP_TEQ_(S_LSL_IMM); }
static int OP_TEQ_LSL_REG(const u32 i) { OP_TEQ_(S_LSL_REG); }
static int OP_TEQ_LSR_IMM(const u32 i) { OP_TEQ_(S_LSR_IMM); }
static int OP_TEQ_LSR_REG(const u32 i) { OP_TEQ_(S_LSR_REG); }
static int OP_TEQ_ASR_IMM(const u32 i) { OP_TEQ_(S_ASR_IMM); }
static int OP_TEQ_ASR_REG(const u32 i) { OP_TEQ_(S_ASR_REG); }
static int OP_TEQ_ROR_IMM(const u32 i) { OP_TEQ_(S_ROR_IMM); }
static int OP_TEQ_ROR_REG(const u32 i) { OP_TEQ_(S_ROR_REG); }
static int OP_TEQ_IMM_VAL(const u32 i) { OP_TEQ_(S_IMM_VAL); }

//-----------------------------------------------------------------------------
//   CMP
//-----------------------------------------------------------------------------
#define OP_CMP(arg) \
	DEBUG_DYNARC(tst_teq_cmp_cmn) \
	arg; \
	c.CMP(map_reg(16), rhs); \
	regman.free_temp32(rhs); \
	SET_NZCV(1); \
	return 1;

static int OP_CMP_LSL_IMM(const u32 i) { OP_CMP(LSL_IMM); }
static int OP_CMP_LSL_REG(const u32 i) { OP_CMP(LSL_REG); }
static int OP_CMP_LSR_IMM(const u32 i) { OP_CMP(LSR_IMM); }
static int OP_CMP_LSR_REG(const u32 i) { OP_CMP(LSR_REG); }
static int OP_CMP_ASR_IMM(const u32 i) { OP_CMP(ASR_IMM); }
static int OP_CMP_ASR_REG(const u32 i) { OP_CMP(ASR_REG); }
static int OP_CMP_ROR_IMM(const u32 i) { OP_CMP(ROR_IMM); }
static int OP_CMP_ROR_REG(const u32 i) { OP_CMP(ROR_REG); }
static int OP_CMP_IMM_VAL(const u32 i) { OP_CMP(IMM_VAL); }
#undef OP_CMP

//-----------------------------------------------------------------------------
//   CMN
//-----------------------------------------------------------------------------
#define OP_CMN(arg) \
	DEBUG_DYNARC(tst_teq_cmp_cmn) \
	arg; \
	c.CMN(map_reg(16), rhs); \
	regman.free_temp32(rhs); \
	SET_NZCV \
	return 1;

static int OP_CMN_LSL_IMM(const u32 i) { OP_CMN(LSL_IMM); }
static int OP_CMN_LSL_REG(const u32 i) { OP_CMN(LSL_REG); }
static int OP_CMN_LSR_IMM(const u32 i) { OP_CMN(LSR_IMM); }
static int OP_CMN_LSR_REG(const u32 i) { OP_CMN(LSR_REG); }
static int OP_CMN_ASR_IMM(const u32 i) { OP_CMN(ASR_IMM); }
static int OP_CMN_ASR_REG(const u32 i) { OP_CMN(ASR_REG); }
static int OP_CMN_ROR_IMM(const u32 i) { OP_CMN(ROR_IMM); }
static int OP_CMN_ROR_REG(const u32 i) { OP_CMN(ROR_REG); }
static int OP_CMN_IMM_VAL(const u32 i) { OP_CMN(IMM_VAL); }
#undef OP_CMN

//-----------------------------------------------------------------------------
//   MOV
//-----------------------------------------------------------------------------
#define OP_MOV(arg, a64inst) \
	DEBUG_DYNARC(mov) \
    arg; \
	c.a64inst(map_reg(12), rhs); \
	regman.free_temp32(rhs); \
	if(REG_POS(i,12)==15) \
	{ \
		c.STR(INDEX_UNSIGNED, map_reg(15), RCPU, offsetof(armcpu_t, next_instruction)); \
		return 1; \
	} \
    return 1;

static int OP_MOV_LSL_IMM(const u32 i) { if (i == 0xE1A00000) { /* nop */ JIT_COMMENT("nop"); return 1; } OP_MOV(LSL_IMM, MOV); }
static int OP_MOV_LSL_REG(const u32 i) { OP_MOV(LSL_REG; if (REG_POS(i,0) == 15) c.ADD(rhs, rhs, 4);, MOV); }
static int OP_MOV_LSR_IMM(const u32 i) { OP_MOV(LSR_IMM, MOV); }
static int OP_MOV_LSR_REG(const u32 i) { OP_MOV(LSR_REG; if (REG_POS(i,0) == 15) c.ADD(rhs, rhs, 4);, MOV); }
static int OP_MOV_ASR_IMM(const u32 i) { OP_MOV(ASR_IMM, MOV); }
static int OP_MOV_ASR_REG(const u32 i) { OP_MOV(ASR_REG, MOV); }
static int OP_MOV_ROR_IMM(const u32 i) { OP_MOV(ROR_IMM, MOV); }
static int OP_MOV_ROR_REG(const u32 i) { OP_MOV(ROR_REG, MOV); }
static int OP_MOV_IMM_VAL(const u32 i) { OP_MOV(IMM_VAL, MOV); }

#define OP_MOV_S(arg, a64inst) \
	DEBUG_DYNARC(mov) \
    arg; \
	c.MOV(map_reg(12), rhs); \
	if(REG_POS(i,12)==15) \
	{ \
		S_DST_R15; \
		c.ADD(Rtotal_cycles, Rtotal_cycles, 2); \
		return 1; \
	} \
	c.TST(map_reg(12), map_reg(12)); \
	SET_NZC; \
    return 1;

static int OP_MOV_S_LSL_IMM(const u32 i) { OP_MOV_S(S_LSL_IMM, MOV); }
static int OP_MOV_S_LSL_REG(const u32 i) { OP_MOV_S(S_LSL_REG; if (REG_POS(i,0) == 15) c.ADD(rhs, rhs, 4);, MOV); }
static int OP_MOV_S_LSR_IMM(const u32 i) { OP_MOV_S(S_LSR_IMM, MOV); }
static int OP_MOV_S_LSR_REG(const u32 i) { OP_MOV_S(S_LSR_REG; if (REG_POS(i,0) == 15) c.ADD(rhs, rhs, 4);, MOV); }
static int OP_MOV_S_ASR_IMM(const u32 i) { OP_MOV_S(S_ASR_IMM, MOV); }
static int OP_MOV_S_ASR_REG(const u32 i) { OP_MOV_S(S_ASR_REG, MOV); }
static int OP_MOV_S_ROR_IMM(const u32 i) { OP_MOV_S(S_ROR_IMM, MOV); }
static int OP_MOV_S_ROR_REG(const u32 i) { OP_MOV_S(S_ROR_REG, MOV); }
static int OP_MOV_S_IMM_VAL(const u32 i) { OP_MOV_S(S_IMM_VAL, MOV); }

//-----------------------------------------------------------------------------
//   MVN
//-----------------------------------------------------------------------------
static int OP_MVN_LSL_IMM(const u32 i) { OP_MOV(LSL_IMM, MVN); }
static int OP_MVN_LSL_REG(const u32 i) { OP_MOV(LSL_REG, MVN); }
static int OP_MVN_LSR_IMM(const u32 i) { OP_MOV(LSR_IMM, MVN); }
static int OP_MVN_LSR_REG(const u32 i) { OP_MOV(LSR_REG, MVN); }
static int OP_MVN_ASR_IMM(const u32 i) { OP_MOV(ASR_IMM, MVN); }
static int OP_MVN_ASR_REG(const u32 i) { OP_MOV(ASR_REG, MVN); }
static int OP_MVN_ROR_IMM(const u32 i) { OP_MOV(ROR_IMM, MVN); }
static int OP_MVN_ROR_REG(const u32 i) { OP_MOV(ROR_REG, MVN); }
static int OP_MVN_IMM_VAL(const u32 i) { OP_MOV(IMM_VAL, MVN); }

static int OP_MVN_S_LSL_IMM(const u32 i) { OP_MOV_S(S_LSL_IMM, MVN); }
static int OP_MVN_S_LSL_REG(const u32 i) { OP_MOV_S(S_LSL_REG, MVN); }
static int OP_MVN_S_LSR_IMM(const u32 i) { OP_MOV_S(S_LSR_IMM, MVN); }
static int OP_MVN_S_LSR_REG(const u32 i) { OP_MOV_S(S_LSR_REG, MVN); }
static int OP_MVN_S_ASR_IMM(const u32 i) { OP_MOV_S(S_ASR_IMM, MVN); }
static int OP_MVN_S_ASR_REG(const u32 i) { OP_MOV_S(S_ASR_REG, MVN); }
static int OP_MVN_S_ROR_IMM(const u32 i) { OP_MOV_S(S_ROR_IMM, MVN); }
static int OP_MVN_S_ROR_REG(const u32 i) { OP_MOV_S(S_ROR_REG, MVN); }
static int OP_MVN_S_IMM_VAL(const u32 i) { OP_MOV_S(S_IMM_VAL, MVN); }

//-----------------------------------------------------------------------------
//   QADD / QDADD / QSUB / QDSUB
//-----------------------------------------------------------------------------
// TODO
static int OP_QADD(const u32 i) { printf("JIT: unimplemented OP_QADD\n"); return 0; }
static int OP_QSUB(const u32 i) { printf("JIT: unimplemented OP_QSUB\n"); return 0; }
static int OP_QDADD(const u32 i) { printf("JIT: unimplemented OP_QDADD\n"); return 0; }
static int OP_QDSUB(const u32 i) { printf("JIT: unimplemented OP_QDSUB\n"); return 0; }

//-----------------------------------------------------------------------------
//   MUL
//-----------------------------------------------------------------------------
static void MUL_Mxx_END(ARM64Reg x, bool sign, int cycles)
{
	auto y = regman.alloc_temp32();
	if(sign)
	{
		c.CLS(y, x);
		c.MOV(y, y, ArithOption(y, ST_LSR, 3));
	}
	else
	{
		c.CLZ(y, x);
		c.MOV(y, y, ArithOption(y, ST_LSR, 3));
	}
	c.ORRI2R(y, y, 1);
	c.ADD(Rcycles, y, cycles+1);
	regman.free_temp32(y);
}

#define OP_MUL_(width, sign, accum, flags) \
	DEBUG_DYNARC(mul) \
	auto res = regman.alloc_temp64(); \
	ARM64Reg ra; \
	if (accum) { \
		if (width) \
		{ \
			ra = regman.alloc_temp64(); \
			c.MOV(ra, map_reg(12)); \
			c.MOV(ra, map_reg(16), ArithOption(ra, ST_LSL, 32)); \
		} \
		else \
			ra = regman.map_reg64(REG_POS(i, 12)); \
	} \
	if (sign) \
		c.SMADDL(res, map_reg(0), map_reg(8), accum ? ra : ZR); \
	else \
		c.UMADDL(res, map_reg(0), map_reg(8), accum ? ra : ZR); \
	if (flags) \
	{ \
		c.TST(res, res); \
		SET_NZ(0) \
	} \
	if (!width) \
		c.MOV(map_reg(16), res); \
	else \
	{ \
		c.MOV(map_reg(12), res); \
		c.MOV(map_reg(16), res, ArithOption(map_reg(16), ST_LSR, 32)); \
		if (accum) regman.free_temp64(ra); \
	} \
	regman.free_temp64(res); \
	MUL_Mxx_END(map_reg(8), sign, 1+width+accum); \
	return 1;

static int OP_MUL(const u32 i) { OP_MUL_(0, 1, 0, 0); }
static int OP_MLA(const u32 i) { OP_MUL_(0, 1, 1, 0); }
static int OP_UMULL(const u32 i) { OP_MUL_(1, 0, 0, 0); }
static int OP_UMLAL(const u32 i) { OP_MUL_(1, 0, 1, 0); }
static int OP_SMULL(const u32 i) { OP_MUL_(1, 1, 0, 0); }
static int OP_SMLAL(const u32 i) { OP_MUL_(1, 1, 1, 0); }

static int OP_MUL_S(const u32 i) { OP_MUL_(0, 1, 0, 1); }
static int OP_MLA_S(const u32 i) { OP_MUL_(0, 1, 1, 1); }
static int OP_UMULL_S(const u32 i) { OP_MUL_(1, 0, 0, 1); }
static int OP_UMLAL_S(const u32 i) { OP_MUL_(1, 0, 1, 1); }
static int OP_SMULL_S(const u32 i) { OP_MUL_(1, 1, 0, 1); }
static int OP_SMLAL_S(const u32 i) { OP_MUL_(1, 1, 1, 1); }

#define OP_MULxy_(x, y, width, accum, flags) \
	DEBUG_DYNARC(mul) \
	auto lhs = regman.alloc_temp32(); \
	auto rhs = regman.alloc_temp32(); \
	mov_reg_##x(lhs, 0); \
	mov_reg_##y(rhs, 8); \
	auto res = regman.alloc_temp64(); \
	ARM64Reg ra; \
	if (accum) { \
		if (width) \
		{ \
			ra = regman.alloc_temp64(); \
			c.MOV(ra, map_reg(12)); \
			c.MOV(ra, map_reg(16), ArithOption(ra, ST_LSL, 32)); \
		} \
		else \
			ra = regman.map_reg64(REG_POS(i, 12)); \
	} \
	if (!flags || width) \
		c.SMADDL(res, lhs, rhs, accum ? ra : ZR); \
	else \
	{ \
		c.SMULL(res, lhs, rhs); \
		c.ADDS(map_reg(16), res, ra); \
		SET_Q \
	} \
	regman.free_temp32(lhs); \
	regman.free_temp32(rhs); \
	if (width) \
	{ \
		c.MOV(map_reg(12), res); \
		c.MOV(map_reg(16), res, ArithOption(map_reg(16), ST_LSR, 32)); \
		if (accum) regman.free_temp64(ra); \
	} \
	regman.free_temp64(res); \
	return 1;


//-----------------------------------------------------------------------------
//   SMUL
//-----------------------------------------------------------------------------
static int OP_SMUL_B_B(const u32 i) { OP_MULxy_(L, L, 0, 0, 0); }
static int OP_SMUL_B_T(const u32 i) { OP_MULxy_(L, H, 0, 0, 0); }
static int OP_SMUL_T_B(const u32 i) { OP_MULxy_(H, L, 0, 0, 0); }
static int OP_SMUL_T_T(const u32 i) { OP_MULxy_(H, H, 0, 0, 0); }

//-----------------------------------------------------------------------------
//   SMLA
//-----------------------------------------------------------------------------
static int OP_SMLA_B_B(const u32 i) { OP_MULxy_(L, L, 0, 1, 1); }
static int OP_SMLA_B_T(const u32 i) { OP_MULxy_(L, H, 0, 1, 1); }
static int OP_SMLA_T_B(const u32 i) { OP_MULxy_(H, L, 0, 1, 1); }
static int OP_SMLA_T_T(const u32 i) { OP_MULxy_(H, H, 0, 1, 1); }

//-----------------------------------------------------------------------------
//   SMLAL
//-----------------------------------------------------------------------------
static int OP_SMLAL_B_B(const u32 i) { OP_MULxy_(L, L, 1, 1, 1); }
static int OP_SMLAL_B_T(const u32 i) { OP_MULxy_(L, H, 1, 1, 1); }
static int OP_SMLAL_T_B(const u32 i) { OP_MULxy_(H, L, 1, 1, 1); }
static int OP_SMLAL_T_T(const u32 i) { OP_MULxy_(H, H, 1, 1, 1); }

//-----------------------------------------------------------------------------
//   SMULW / SMLAW
//-----------------------------------------------------------------------------
#ifdef ASMJIT_X64
#define OP_SMxxW_(x, accum, flags) \
	GpVar lhs = c.newGpVar(kX86VarTypeGpz); \
	GpVar rhs = c.newGpVar(kX86VarTypeGpz); \
	c.movsx(lhs, reg_pos_ptr##x(8)); \
	c.movsxd(rhs, reg_pos_ptr(0)); \
	c.imul(lhs, rhs);  \
	c.sar(lhs, 16); \
	if (accum) c.add(lhs, reg_pos_ptr(12)); \
	c.mov(reg_pos_ptr(16), lhs.r32()); \
	if (flags) { SET_Q; } \
	return 1;
#else
#define OP_SMxxW_(x, accum, flags) \
	DEBUG_DYNARC(mul) \
	auto rhs = regman.alloc_temp32(); \
	auto res = regman.alloc_temp64(); \
	mov_reg_##x(rhs, 8); \
	c.SMULL(res, map_reg(0), rhs); \
	c.ASR(map_reg(16), res, 15); \
	if (accum) \
	{ \
		if (flags) \
		{ \
			c.ADDS(map_reg(16), map_reg(16), map_reg(12)); \
			SET_Q \
		} \
		else \
			c.ADD(map_reg(16), map_reg(16), map_reg(12)); \
	} \
	regman.free_temp32(rhs); \
	regman.free_temp64(res); \
	return 1;
#endif

static int OP_SMULW_B(const u32 i) { OP_SMxxW_(L, 0, 0); }
static int OP_SMULW_T(const u32 i) { OP_SMxxW_(H, 0, 0); }
static int OP_SMLAW_B(const u32 i) { OP_SMxxW_(L, 1, 1); }
static int OP_SMLAW_T(const u32 i) { OP_SMxxW_(H, 1, 1); }

//-----------------------------------------------------------------------------
//   MRS / MSR
//-----------------------------------------------------------------------------
static int OP_MRS_CPSR(const u32 i)
{
	DEBUG_DYNARC(mrs)
	c.MOV(map_reg(12), RCPSR);
	return 1;
}

static int OP_MRS_SPSR(const u32 i)
{
	DEBUG_DYNARC(mrs)
	c.STR(INDEX_UNSIGNED, map_reg(12), RCPU, offsetof(armcpu_t, SPSR.val));
	return 1;
}

// TODO: SPSR: if(cpu->CPSR.bits.mode == USR || cpu->CPSR.bits.mode == SYS) return 1;
#define OP_MSR_(reg, args, sw) \
	DEBUG_DYNARC(mrs) \
	args; \
	switch (((i>>16) & 0xF)) \
	{ \
		case 0x1:		/* bit 16 */ \
			{ \
				auto mode = regman.alloc_temp32(); \
				c.ANDI2R(mode, RCPSR, 0x1F); \
				c.CMPI2R(mode, USR); \
				auto __skip = c.B(CC_EQ); \
				if (sw) \
				{ \
					c.MOV(X0, RCPU); \
					c.ANDI2R(W1, rhs, 0x1F); \
					regman.call(X2, armcpu_switchMode, true); \
				} \
				c.STRB(INDEX_UNSIGNED, rhs, RCPU, offsetof(armcpu_t, reg)); \
				changeCPSR \
				regman.load_cpsr(); \
				c.SetJumpTarget(__skip); \
				regman.free_temp32(mode); \
				regman.free_temp32(rhs); \
			} \
			return 1; \
		case 0x2:		/* bit 17 */ \
			{ \
				regman.save_cpsr(); \
				auto mode = regman.alloc_temp32(); \
				c.ANDI2R(mode, RCPSR, 0x1F); \
				c.CMPI2R(mode, USR); \
				auto __skip = c.B(CC_EQ); \
				c.LSR(rhs, rhs, 8); \
				c.STRB(INDEX_UNSIGNED, rhs, RCPU, offsetof(armcpu_t, reg) + 1); \
				changeCPSR; \
				regman.load_cpsr(); \
				c.SetJumpTarget(__skip); \
				regman.free_temp32(mode); \
				regman.free_temp32(rhs); \
			} \
			return 1; \
		case 0x4:		/* bit 18 */ \
			{ \
				regman.save_cpsr(); \
				auto mode = regman.alloc_temp32(); \
				c.ANDI2R(mode, RCPSR, 0x1F); \
				c.CMPI2R(mode, USR); \
				auto __skip = c.B(CC_EQ); \
				c.LSR(rhs, rhs, 16); \
				c.STRB(INDEX_UNSIGNED, rhs, RCPU, offsetof(armcpu_t, reg) + 2); \
				changeCPSR; \
				regman.load_cpsr(); \
				c.SetJumpTarget(__skip); \
				regman.free_temp32(mode); \
				regman.free_temp32(rhs); \
			} \
			return 1; \
		case 0x8:		/* bit 19 */ \
			{ \
				regman.save_cpsr(); \
				c.LSR(rhs, rhs, 24); \
				c.STRB(INDEX_UNSIGNED, rhs, RCPU, offsetof(armcpu_t, reg) + 3); \
				changeCPSR; \
				regman.load_cpsr(); \
				regman.free_temp32(rhs); \
			} \
			return 1; \
		default: \
			break; \
	} \
\
	static u32 byte_mask =	(BIT16(i)?0x000000FF:0x00000000) | \
							(BIT17(i)?0x0000FF00:0x00000000) | \
							(BIT18(i)?0x00FF0000:0x00000000) | \
							(BIT19(i)?0xFF000000:0x00000000); \
	static u32 byte_mask_USR = (BIT19(i)?0xFF000000:0x00000000); \
\
	auto mode = regman.alloc_temp32(); \
	c.ANDI2R(mode, RCPSR, 0x1f); \
	c.CMPI2R(mode, USR); \
	regman.free_temp32(mode); \
	auto __USR = c.B(CC_EQ); \
	if (sw && BIT16(i)) \
	{ \
		c.MOV(X0, RCPU); \
		c.ANDI2R(W1, rhs, 0x1f); \
		regman.call(X2, armcpu_switchMode, true); \
	} \
	c.ANDI2R(rhs, rhs, byte_mask); \
	c.ANDI2R(RCPSR, RCPSR, ~byte_mask); \
	c.ORR(RCPSR, RCPSR, rhs); \
	auto __done = c.B(); \
	c.SetJumpTarget(__USR); \
	c.ANDI2R(rhs, rhs, byte_mask_USR); \
	c.ANDI2R(RCPSR, RCPSR, ~byte_mask_USR); \
	c.ORR(RCPSR, RCPSR, rhs); \
	c.SetJumpTarget(__done); \
	regman.free_temp32(rhs); \
	changeCPSR; \
	return 1;

static int OP_MSR_CPSR(const u32 i) { OP_MSR_(CPSR, REG_OFF, 1); }
static int OP_MSR_SPSR(const u32 i) { OP_MSR_(SPSR, REG_OFF, 0); }
static int OP_MSR_CPSR_IMM_VAL(const u32 i) { OP_MSR_(CPSR, IMM_VAL, 1); }
static int OP_MSR_SPSR_IMM_VAL(const u32 i) { OP_MSR_(SPSR, IMM_VAL, 0); }

//-----------------------------------------------------------------------------
//   LDR
//-----------------------------------------------------------------------------
typedef u32 (FASTCALL* OpLDR)(u32, u32*);

// 98% of all memory accesses land in the same region as the first execution of
// that instruction, so keep multiple copies with different fastpaths.
// The copies don't need to differ in any way; the point is merely to cooperate
// with x86 branch prediction.

enum {
	MEMTYPE_GENERIC = 0, // no assumptions
	MEMTYPE_MAIN = 1,
	MEMTYPE_DTCM = 2,
	MEMTYPE_ERAM = 3,
	MEMTYPE_SWIRAM = 4,
	MEMTYPE_OTHER = 5, // memory that is known to not be MAIN, DTCM, ERAM, or SWIRAM
};

static u32 classify_adr(u32 adr, bool store)
{
	if(PROCNUM==ARMCPU_ARM9 && (adr & ~0x3FFF) == MMU.DTCMRegion)
		return MEMTYPE_DTCM;
	else if((adr & 0x0F000000) == 0x02000000)
		return MEMTYPE_MAIN;
	else if(PROCNUM==ARMCPU_ARM7 && !store && (adr & 0xFF800000) == 0x03800000)
		return MEMTYPE_ERAM;
	else if(PROCNUM==ARMCPU_ARM7 && !store && (adr & 0xFF800000) == 0x03000000)
		return MEMTYPE_SWIRAM;
	else
		return MEMTYPE_GENERIC;
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_LDR(u32 adr, u32 *dstreg)
{
	u32 data = READ32(cpu->mem_if->data, adr);
	if(adr&3)
		data = ROR(data, 8*(adr&3));
	*dstreg = data;
	return MMU_aluMemAccessCycles<PROCNUM,32,MMU_AD_READ>(3,adr);
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_LDRH(u32 adr, u32 *dstreg)
{
	*dstreg = READ16(cpu->mem_if->data, adr);
	return MMU_aluMemAccessCycles<PROCNUM,16,MMU_AD_READ>(3,adr);
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_LDRSH(u32 adr, u32 *dstreg)
{
	*dstreg = (s16)READ16(cpu->mem_if->data, adr);
	return MMU_aluMemAccessCycles<PROCNUM,16,MMU_AD_READ>(3,adr);
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_LDRB(u32 adr, u32 *dstreg)
{
	*dstreg = READ8(cpu->mem_if->data, adr);
	return MMU_aluMemAccessCycles<PROCNUM,8,MMU_AD_READ>(3,adr);
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_LDRSB(u32 adr, u32 *dstreg)
{
	*dstreg = (s8)READ8(cpu->mem_if->data, adr);
	return MMU_aluMemAccessCycles<PROCNUM,8,MMU_AD_READ>(3,adr);
}

#define T(op) op<0,0>, op<0,1>, op<0,2>, NULL, NULL, op<1,0>, op<1,1>, NULL, op<1,3>, op<1,4>
static const OpLDR LDR_tab[2][5]   = { T(OP_LDR) };
static const OpLDR LDRH_tab[2][5]  = { T(OP_LDRH) };
static const OpLDR LDRSH_tab[2][5] = { T(OP_LDRSH) };
static const OpLDR LDRB_tab[2][5]  = { T(OP_LDRB) };
static const OpLDR LDRSB_tab[2][5]  = { T(OP_LDRSB) };
#undef T

static u32 ADD(u32 lhs, u32 rhs) { return lhs + rhs; }
static u32 SUB(u32 lhs, u32 rhs) { return lhs - rhs; }

#define OP_LDR_(mem_op, arg, sign_op, writeback) \
	DEBUG_DYNARC(ldr) \
	c.MOV(W0, map_reg(16)); \
	arg; \
	if (writeback == 0) \
		c.sign_op(W0, W0, rhs); \
	else if (writeback < 0) \
	{ \
		c.sign_op(W0, W0, rhs); \
		c.MOV(map_reg(16), W0); \
	} \
	else if (writeback > 0) \
	{ \
		c.sign_op(map_reg(16), W0, rhs); \
	} \
	regman.free_temp32(rhs); \
	u32 adr_first = sign_op(cpu->R[REG_POS(i,16)], rhs_first); \
	c.ADD(X1, RCPU, offsetof(armcpu_t, R) + REG_POS(i,12)); \
	regman.call(X2, mem_op##_tab[PROCNUM][classify_adr(adr_first,0)], true); \
	c.MOV(Rcycles, W0); \
	c.LDR(INDEX_UNSIGNED, map_reg(12), RCPU, offsetof(armcpu_t, R) + REG_POS(i,12)); \
	if(REG_POS(i,12)==15) \
	{ \
		auto tmp = regman.alloc_temp32(); \
		if (PROCNUM == 0) \
		{ \
			c.UBFX(tmp, regman.map_reg32(15), 0, 1); \
			c.ORR(RCPSR, RCPSR, tmp, ArithOption(tmp, ST_LSR, 5)); \
			c.ANDI2R(tmp, regman.map_reg32(15), 0xFFFFFFFE); \
		} \
		else \
			c.ANDI2R(tmp, regman.map_reg32(15), 0xFFFFFFFC); \
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction)); \
		regman.free_temp32(tmp); \
	} \
	return 1;

// LDR
static int OP_LDR_P_IMM_OFF(const u32 i) { OP_LDR_(LDR, IMM_OFF_12, ADD, 0); }
static int OP_LDR_M_IMM_OFF(const u32 i) { OP_LDR_(LDR, IMM_OFF_12, SUB, 0); }
static int OP_LDR_P_LSL_IMM_OFF(const u32 i) { OP_LDR_(LDR, LSL_IMM, ADD, 0); }
static int OP_LDR_M_LSL_IMM_OFF(const u32 i) { OP_LDR_(LDR, LSL_IMM, SUB, 0); }
static int OP_LDR_P_LSR_IMM_OFF(const u32 i) { OP_LDR_(LDR, LSR_IMM, ADD, 0); }
static int OP_LDR_M_LSR_IMM_OFF(const u32 i) { OP_LDR_(LDR, LSR_IMM, SUB, 0); }
static int OP_LDR_P_ASR_IMM_OFF(const u32 i) { OP_LDR_(LDR, ASR_IMM, ADD, 0); }
static int OP_LDR_M_ASR_IMM_OFF(const u32 i) { OP_LDR_(LDR, ASR_IMM, SUB, 0); }
static int OP_LDR_P_ROR_IMM_OFF(const u32 i) { OP_LDR_(LDR, ROR_IMM, ADD, 0); }
static int OP_LDR_M_ROR_IMM_OFF(const u32 i) { OP_LDR_(LDR, ROR_IMM, SUB, 0); }

static int OP_LDR_P_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, IMM_OFF_12, ADD, -1); }
static int OP_LDR_M_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, IMM_OFF_12, SUB, -1); }
static int OP_LDR_P_LSL_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, LSL_IMM, ADD, -1); }
static int OP_LDR_M_LSL_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, LSL_IMM, SUB, -1); }
static int OP_LDR_P_LSR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, LSR_IMM, ADD, -1); }
static int OP_LDR_M_LSR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, LSR_IMM, SUB, -1); }
static int OP_LDR_P_ASR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, ASR_IMM, ADD, -1); }
static int OP_LDR_M_ASR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, ASR_IMM, SUB, -1); }
static int OP_LDR_P_ROR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, ROR_IMM, ADD, -1); }
static int OP_LDR_M_ROR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDR, ROR_IMM, SUB, -1); }
static int OP_LDR_P_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, IMM_OFF_12, ADD, 1); }
static int OP_LDR_M_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, IMM_OFF_12, SUB, 1); }
static int OP_LDR_P_LSL_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, LSL_IMM, ADD, 1); }
static int OP_LDR_M_LSL_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, LSL_IMM, SUB, 1); }
static int OP_LDR_P_LSR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, LSR_IMM, ADD, 1); }
static int OP_LDR_M_LSR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, LSR_IMM, SUB, 1); }
static int OP_LDR_P_ASR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, ASR_IMM, ADD, 1); }
static int OP_LDR_M_ASR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, ASR_IMM, SUB, 1); }
static int OP_LDR_P_ROR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, ROR_IMM, ADD, 1); }
static int OP_LDR_M_ROR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDR, ROR_IMM, SUB, 1); }

// LDRH
static int OP_LDRH_P_IMM_OFF(const u32 i) { OP_LDR_(LDRH, IMM_OFF, ADD, 0); }
static int OP_LDRH_M_IMM_OFF(const u32 i) { OP_LDR_(LDRH, IMM_OFF, SUB, 0); }
static int OP_LDRH_P_REG_OFF(const u32 i) { OP_LDR_(LDRH, REG_OFF, ADD, 0); }
static int OP_LDRH_M_REG_OFF(const u32 i) { OP_LDR_(LDRH, REG_OFF, SUB, 0); }

static int OP_LDRH_PRE_INDE_P_IMM_OFF(const u32 i) { OP_LDR_(LDRH, IMM_OFF, ADD, -1); }
static int OP_LDRH_PRE_INDE_M_IMM_OFF(const u32 i) { OP_LDR_(LDRH, IMM_OFF, SUB, -1); }
static int OP_LDRH_PRE_INDE_P_REG_OFF(const u32 i) { OP_LDR_(LDRH, REG_OFF, ADD, -1); }
static int OP_LDRH_PRE_INDE_M_REG_OFF(const u32 i) { OP_LDR_(LDRH, REG_OFF, SUB, -1); }
static int OP_LDRH_POS_INDE_P_IMM_OFF(const u32 i) { OP_LDR_(LDRH, IMM_OFF, ADD, 1); }
static int OP_LDRH_POS_INDE_M_IMM_OFF(const u32 i) { OP_LDR_(LDRH, IMM_OFF, SUB, 1); }
static int OP_LDRH_POS_INDE_P_REG_OFF(const u32 i) { OP_LDR_(LDRH, REG_OFF, ADD, 1); }
static int OP_LDRH_POS_INDE_M_REG_OFF(const u32 i) { OP_LDR_(LDRH, REG_OFF, SUB, 1); }

// LDRSH
static int OP_LDRSH_P_IMM_OFF(const u32 i) { OP_LDR_(LDRSH, IMM_OFF, ADD, 0); }
static int OP_LDRSH_M_IMM_OFF(const u32 i) { OP_LDR_(LDRSH, IMM_OFF, SUB, 0); }
static int OP_LDRSH_P_REG_OFF(const u32 i) { OP_LDR_(LDRSH, REG_OFF, ADD, 0); }
static int OP_LDRSH_M_REG_OFF(const u32 i) { OP_LDR_(LDRSH, REG_OFF, SUB, 0); }

static int OP_LDRSH_PRE_INDE_P_IMM_OFF(const u32 i) { OP_LDR_(LDRSH, IMM_OFF, ADD, -1); }
static int OP_LDRSH_PRE_INDE_M_IMM_OFF(const u32 i) { OP_LDR_(LDRSH, IMM_OFF, SUB, -1); }
static int OP_LDRSH_PRE_INDE_P_REG_OFF(const u32 i) { OP_LDR_(LDRSH, REG_OFF, ADD, -1); }
static int OP_LDRSH_PRE_INDE_M_REG_OFF(const u32 i) { OP_LDR_(LDRSH, REG_OFF, SUB, -1); }
static int OP_LDRSH_POS_INDE_P_IMM_OFF(const u32 i) { OP_LDR_(LDRSH, IMM_OFF, ADD, 1); }
static int OP_LDRSH_POS_INDE_M_IMM_OFF(const u32 i) { OP_LDR_(LDRSH, IMM_OFF, SUB, 1); }
static int OP_LDRSH_POS_INDE_P_REG_OFF(const u32 i) { OP_LDR_(LDRSH, REG_OFF, ADD, 1); }
static int OP_LDRSH_POS_INDE_M_REG_OFF(const u32 i) { OP_LDR_(LDRSH, REG_OFF, SUB, 1); }

// LDRB
static int OP_LDRB_P_IMM_OFF(const u32 i) { OP_LDR_(LDRB, IMM_OFF_12, ADD, 0); }
static int OP_LDRB_M_IMM_OFF(const u32 i) { OP_LDR_(LDRB, IMM_OFF_12, SUB, 0); }
static int OP_LDRB_P_LSL_IMM_OFF(const u32 i) { OP_LDR_(LDRB, LSL_IMM, ADD, 0); }
static int OP_LDRB_M_LSL_IMM_OFF(const u32 i) { OP_LDR_(LDRB, LSL_IMM, SUB, 0); }
static int OP_LDRB_P_LSR_IMM_OFF(const u32 i) { OP_LDR_(LDRB, LSR_IMM, ADD, 0); }
static int OP_LDRB_M_LSR_IMM_OFF(const u32 i) { OP_LDR_(LDRB, LSR_IMM, SUB, 0); }
static int OP_LDRB_P_ASR_IMM_OFF(const u32 i) { OP_LDR_(LDRB, ASR_IMM, ADD, 0); }
static int OP_LDRB_M_ASR_IMM_OFF(const u32 i) { OP_LDR_(LDRB, ASR_IMM, SUB, 0); }
static int OP_LDRB_P_ROR_IMM_OFF(const u32 i) { OP_LDR_(LDRB, ROR_IMM, ADD, 0); }
static int OP_LDRB_M_ROR_IMM_OFF(const u32 i) { OP_LDR_(LDRB, ROR_IMM, SUB, 0); }

static int OP_LDRB_P_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, IMM_OFF_12, ADD, -1); }
static int OP_LDRB_M_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, IMM_OFF_12, SUB, -1); }
static int OP_LDRB_P_LSL_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, LSL_IMM, ADD, -1); }
static int OP_LDRB_M_LSL_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, LSL_IMM, SUB, -1); }
static int OP_LDRB_P_LSR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, LSR_IMM, ADD, -1); }
static int OP_LDRB_M_LSR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, LSR_IMM, SUB, -1); }
static int OP_LDRB_P_ASR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, ASR_IMM, ADD, -1); }
static int OP_LDRB_M_ASR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, ASR_IMM, SUB, -1); }
static int OP_LDRB_P_ROR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, ROR_IMM, ADD, -1); }
static int OP_LDRB_M_ROR_IMM_OFF_PREIND(const u32 i) { OP_LDR_(LDRB, ROR_IMM, SUB, -1); }
static int OP_LDRB_P_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, IMM_OFF_12, ADD, 1); }
static int OP_LDRB_M_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, IMM_OFF_12, SUB, 1); }
static int OP_LDRB_P_LSL_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, LSL_IMM, ADD, 1); }
static int OP_LDRB_M_LSL_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, LSL_IMM, SUB, 1); }
static int OP_LDRB_P_LSR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, LSR_IMM, ADD, 1); }
static int OP_LDRB_M_LSR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, LSR_IMM, SUB, 1); }
static int OP_LDRB_P_ASR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, ASR_IMM, ADD, 1); }
static int OP_LDRB_M_ASR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, ASR_IMM, SUB, 1); }
static int OP_LDRB_P_ROR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, ROR_IMM, ADD, 1); }
static int OP_LDRB_M_ROR_IMM_OFF_POSTIND(const u32 i) { OP_LDR_(LDRB, ROR_IMM, SUB, 1); }

// LDRSB
static int OP_LDRSB_P_IMM_OFF(const u32 i) { OP_LDR_(LDRSB, IMM_OFF, ADD, 0); }
static int OP_LDRSB_M_IMM_OFF(const u32 i) { OP_LDR_(LDRSB, IMM_OFF, SUB, 0); }
static int OP_LDRSB_P_REG_OFF(const u32 i) { OP_LDR_(LDRSB, REG_OFF, ADD, 0); }
static int OP_LDRSB_M_REG_OFF(const u32 i) { OP_LDR_(LDRSB, REG_OFF, SUB, 0); }

static int OP_LDRSB_PRE_INDE_P_IMM_OFF(const u32 i) { OP_LDR_(LDRSB, IMM_OFF, ADD, -1); }
static int OP_LDRSB_PRE_INDE_M_IMM_OFF(const u32 i) { OP_LDR_(LDRSB, IMM_OFF, SUB, -1); }
static int OP_LDRSB_PRE_INDE_P_REG_OFF(const u32 i) { OP_LDR_(LDRSB, REG_OFF, ADD, -1); }
static int OP_LDRSB_PRE_INDE_M_REG_OFF(const u32 i) { OP_LDR_(LDRSB, REG_OFF, SUB, -1); }
static int OP_LDRSB_POS_INDE_P_IMM_OFF(const u32 i) { OP_LDR_(LDRSB, IMM_OFF, ADD, 1); }
static int OP_LDRSB_POS_INDE_M_IMM_OFF(const u32 i) { OP_LDR_(LDRSB, IMM_OFF, SUB, 1); }
static int OP_LDRSB_POS_INDE_P_REG_OFF(const u32 i) { OP_LDR_(LDRSB, REG_OFF, ADD, 1); }
static int OP_LDRSB_POS_INDE_M_REG_OFF(const u32 i) { OP_LDR_(LDRSB, REG_OFF, SUB, 1); }

//-----------------------------------------------------------------------------
//   STR
//-----------------------------------------------------------------------------
template<int PROCNUM, int memtype>
static u32 FASTCALL OP_STR(u32 adr, u32 data)
{
	WRITE32(cpu->mem_if->data, adr, data);
	return MMU_aluMemAccessCycles<PROCNUM,32,MMU_AD_WRITE>(2,adr);
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_STRH(u32 adr, u32 data)
{
	WRITE16(cpu->mem_if->data, adr, data);
	return MMU_aluMemAccessCycles<PROCNUM,16,MMU_AD_WRITE>(2,adr);
}

template<int PROCNUM, int memtype>
static u32 FASTCALL OP_STRB(u32 adr, u32 data)
{
	WRITE8(cpu->mem_if->data, adr, data);
	return MMU_aluMemAccessCycles<PROCNUM,8,MMU_AD_WRITE>(2,adr);
}

typedef u32 (FASTCALL* OpSTR)(u32, u32);
#define T(op) op<0,0>, op<0,1>, op<0,2>, op<1,0>, op<1,1>, NULL
static const OpSTR STR_tab[2][3]   = { T(OP_STR) };
static const OpSTR STRH_tab[2][3]  = { T(OP_STRH) };
static const OpSTR STRB_tab[2][3]  = { T(OP_STRB) };
#undef T

#define OP_STR_(mem_op, arg, sign_op, writeback) \
	DEBUG_DYNARC(str) \
	c.MOV(W0, map_reg(16)); \
	arg; \
	if (writeback == 0) \
		c.sign_op(W0, W0, rhs); \
	else if (writeback < 0) \
	{ \
		c.sign_op(W0, W0, rhs); \
		c.MOV(map_reg(16), W0); \
	} \
	else if (writeback > 0) \
	{ \
		c.sign_op(map_reg(16), W0, rhs); \
	} \
	regman.free_temp32(rhs); \
	u32 adr_first = sign_op(cpu->R[REG_POS(i,16)], rhs_first); \
	c.MOV(W1, map_reg(12)); \
	regman.call(X2, mem_op##_tab[PROCNUM][classify_adr(adr_first,1)], true); \
	c.MOV(Rcycles, W0); \
	return 1;

static int OP_STR_P_IMM_OFF(const u32 i) { OP_STR_(STR, IMM_OFF_12, ADD, 0); }
static int OP_STR_M_IMM_OFF(const u32 i) { OP_STR_(STR, IMM_OFF_12, SUB, 0); }
static int OP_STR_P_LSL_IMM_OFF(const u32 i) { OP_STR_(STR, LSL_IMM, ADD, 0); }
static int OP_STR_M_LSL_IMM_OFF(const u32 i) { OP_STR_(STR, LSL_IMM, SUB, 0); }
static int OP_STR_P_LSR_IMM_OFF(const u32 i) { OP_STR_(STR, LSR_IMM, ADD, 0); }
static int OP_STR_M_LSR_IMM_OFF(const u32 i) { OP_STR_(STR, LSR_IMM, SUB, 0); }
static int OP_STR_P_ASR_IMM_OFF(const u32 i) { OP_STR_(STR, ASR_IMM, ADD, 0); }
static int OP_STR_M_ASR_IMM_OFF(const u32 i) { OP_STR_(STR, ASR_IMM, SUB, 0); }
static int OP_STR_P_ROR_IMM_OFF(const u32 i) { OP_STR_(STR, ROR_IMM, ADD, 0); }
static int OP_STR_M_ROR_IMM_OFF(const u32 i) { OP_STR_(STR, ROR_IMM, SUB, 0); }

static int OP_STR_P_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, IMM_OFF_12, ADD, -1); }
static int OP_STR_M_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, IMM_OFF_12, SUB, -1); }
static int OP_STR_P_LSL_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, LSL_IMM, ADD, -1); }
static int OP_STR_M_LSL_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, LSL_IMM, SUB, -1); }
static int OP_STR_P_LSR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, LSR_IMM, ADD, -1); }
static int OP_STR_M_LSR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, LSR_IMM, SUB, -1); }
static int OP_STR_P_ASR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, ASR_IMM, ADD, -1); }
static int OP_STR_M_ASR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, ASR_IMM, SUB, -1); }
static int OP_STR_P_ROR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, ROR_IMM, ADD, -1); }
static int OP_STR_M_ROR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STR, ROR_IMM, SUB, -1); }
static int OP_STR_P_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, IMM_OFF_12, ADD, 1); }
static int OP_STR_M_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, IMM_OFF_12, SUB, 1); }
static int OP_STR_P_LSL_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, LSL_IMM, ADD, 1); }
static int OP_STR_M_LSL_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, LSL_IMM, SUB, 1); }
static int OP_STR_P_LSR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, LSR_IMM, ADD, 1); }
static int OP_STR_M_LSR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, LSR_IMM, SUB, 1); }
static int OP_STR_P_ASR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, ASR_IMM, ADD, 1); }
static int OP_STR_M_ASR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, ASR_IMM, SUB, 1); }
static int OP_STR_P_ROR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, ROR_IMM, ADD, 1); }
static int OP_STR_M_ROR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STR, ROR_IMM, SUB, 1); }

static int OP_STRH_P_IMM_OFF(const u32 i) { OP_STR_(STRH, IMM_OFF, ADD, 0); }
static int OP_STRH_M_IMM_OFF(const u32 i) { OP_STR_(STRH, IMM_OFF, SUB, 0); }
static int OP_STRH_P_REG_OFF(const u32 i) { OP_STR_(STRH, REG_OFF, ADD, 0); }
static int OP_STRH_M_REG_OFF(const u32 i) { OP_STR_(STRH, REG_OFF, SUB, 0); }

static int OP_STRH_PRE_INDE_P_IMM_OFF(const u32 i) { OP_STR_(STRH, IMM_OFF, ADD, -1); }
static int OP_STRH_PRE_INDE_M_IMM_OFF(const u32 i) { OP_STR_(STRH, IMM_OFF, SUB, -1); }
static int OP_STRH_PRE_INDE_P_REG_OFF(const u32 i) { OP_STR_(STRH, REG_OFF, ADD, -1); }
static int OP_STRH_PRE_INDE_M_REG_OFF(const u32 i) { OP_STR_(STRH, REG_OFF, SUB, -1); }
static int OP_STRH_POS_INDE_P_IMM_OFF(const u32 i) { OP_STR_(STRH, IMM_OFF, ADD, 1); }
static int OP_STRH_POS_INDE_M_IMM_OFF(const u32 i) { OP_STR_(STRH, IMM_OFF, SUB, 1); }
static int OP_STRH_POS_INDE_P_REG_OFF(const u32 i) { OP_STR_(STRH, REG_OFF, ADD, 1); }
static int OP_STRH_POS_INDE_M_REG_OFF(const u32 i) { OP_STR_(STRH, REG_OFF, SUB, 1); }

static int OP_STRB_P_IMM_OFF(const u32 i) { OP_STR_(STRB, IMM_OFF_12, ADD, 0); }
static int OP_STRB_M_IMM_OFF(const u32 i) { OP_STR_(STRB, IMM_OFF_12, SUB, 0); }
static int OP_STRB_P_LSL_IMM_OFF(const u32 i) { OP_STR_(STRB, LSL_IMM, ADD, 0); }
static int OP_STRB_M_LSL_IMM_OFF(const u32 i) { OP_STR_(STRB, LSL_IMM, SUB, 0); }
static int OP_STRB_P_LSR_IMM_OFF(const u32 i) { OP_STR_(STRB, LSR_IMM, ADD, 0); }
static int OP_STRB_M_LSR_IMM_OFF(const u32 i) { OP_STR_(STRB, LSR_IMM, SUB, 0); }
static int OP_STRB_P_ASR_IMM_OFF(const u32 i) { OP_STR_(STRB, ASR_IMM, ADD, 0); }
static int OP_STRB_M_ASR_IMM_OFF(const u32 i) { OP_STR_(STRB, ASR_IMM, SUB, 0); }
static int OP_STRB_P_ROR_IMM_OFF(const u32 i) { OP_STR_(STRB, ROR_IMM, ADD, 0); }
static int OP_STRB_M_ROR_IMM_OFF(const u32 i) { OP_STR_(STRB, ROR_IMM, SUB, 0); }

static int OP_STRB_P_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, IMM_OFF_12, ADD, -1); }
static int OP_STRB_M_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, IMM_OFF_12, SUB, -1); }
static int OP_STRB_P_LSL_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, LSL_IMM, ADD, -1); }
static int OP_STRB_M_LSL_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, LSL_IMM, SUB, -1); }
static int OP_STRB_P_LSR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, LSR_IMM, ADD, -1); }
static int OP_STRB_M_LSR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, LSR_IMM, SUB, -1); }
static int OP_STRB_P_ASR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, ASR_IMM, ADD, -1); }
static int OP_STRB_M_ASR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, ASR_IMM, SUB, -1); }
static int OP_STRB_P_ROR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, ROR_IMM, ADD, -1); }
static int OP_STRB_M_ROR_IMM_OFF_PREIND(const u32 i) { OP_STR_(STRB, ROR_IMM, SUB, -1); }
static int OP_STRB_P_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, IMM_OFF_12, ADD, 1); }
static int OP_STRB_M_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, IMM_OFF_12, SUB, 1); }
static int OP_STRB_P_LSL_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, LSL_IMM, ADD, 1); }
static int OP_STRB_M_LSL_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, LSL_IMM, SUB, 1); }
static int OP_STRB_P_LSR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, LSR_IMM, ADD, 1); }
static int OP_STRB_M_LSR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, LSR_IMM, SUB, 1); }
static int OP_STRB_P_ASR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, ASR_IMM, ADD, 1); }
static int OP_STRB_M_ASR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, ASR_IMM, SUB, 1); }
static int OP_STRB_P_ROR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, ROR_IMM, ADD, 1); }
static int OP_STRB_M_ROR_IMM_OFF_POSTIND(const u32 i) { OP_STR_(STRB, ROR_IMM, SUB, 1); }

//-----------------------------------------------------------------------------
//   LDRD / STRD
//-----------------------------------------------------------------------------
typedef u32 FASTCALL (*LDRD_STRD_REG)(u32);
template<int PROCNUM, u8 Rnum>
static u32 FASTCALL OP_LDRD_REG(u32 adr)
{
	cpu->R[Rnum] = READ32(cpu->mem_if->data, adr);
	cpu->R[Rnum+1] = READ32(cpu->mem_if->data, adr+4);
	return (MMU_memAccessCycles<PROCNUM,32,MMU_AD_READ>(adr) + MMU_memAccessCycles<PROCNUM,32,MMU_AD_READ>(adr+4));
}
template<int PROCNUM, u8 Rnum>
static u32 FASTCALL OP_STRD_REG(u32 adr)
{
	WRITE32(cpu->mem_if->data, adr, cpu->R[Rnum]);
	WRITE32(cpu->mem_if->data, adr + 4, cpu->R[Rnum + 1]);
	return (MMU_memAccessCycles<PROCNUM,32,MMU_AD_WRITE>(adr) + MMU_memAccessCycles<PROCNUM,32,MMU_AD_WRITE>(adr+4));
}
#define T(op, proc) op<proc,0>, op<proc,1>, op<proc,2>, op<proc,3>, op<proc,4>, op<proc,5>, op<proc,6>, op<proc,7>, op<proc,8>, op<proc,9>, op<proc,10>, op<proc,11>, op<proc,12>, op<proc,13>, op<proc,14>, op<proc,15>
static const LDRD_STRD_REG op_ldrd_tab[2][16] = { {T(OP_LDRD_REG, 0)}, {T(OP_LDRD_REG, 1)} };
static const LDRD_STRD_REG op_strd_tab[2][16] = { {T(OP_STRD_REG, 0)}, {T(OP_STRD_REG, 1)} };
#undef T

static int OP_LDRD_STRD_POST_INDEX(const u32 i) 
{
	DEBUG_DYNARC(ldrd_strd_post)
	u8 Rd_num = REG_POS(i, 12);
	
	if (Rd_num == 14)
	{
		printf("OP_LDRD_STRD_POST_INDEX: use R14!!!!\n");
		return 0; // TODO: exception
	}
	if (Rd_num & 0x1)
	{
		printf("OP_LDRD_STRD_POST_INDEX: ERROR!!!!\n");
		return 0; // TODO: exception
	}
	
	c.MOV(W0, map_reg(12));
	regman.call(X2, BIT5(i) ? op_strd_tab[PROCNUM][Rd_num] : op_ldrd_tab[PROCNUM][Rd_num], true);
	c.MOV(Rcycles, W0);
	if (!BIT5(i))
		c.LDR(INDEX_UNSIGNED, map_reg(12), RCPU, offsetof(armcpu_t, R) + 4 * Rd_num);

	// I bit - immediate or register
	if (BIT22(i))
	{
		IMM_OFF;
		BIT23(i)?c.ADD(map_reg(16), map_reg(16), rhs):c.SUB(map_reg(16), map_reg(16), rhs);
		regman.free_temp32(rhs);
	}
	else
	{
		BIT23(i)?c.ADD(map_reg(16), map_reg(16), map_reg(0)):c.SUB(map_reg(16), map_reg(16), map_reg(0));
	}
	
	emit_MMU_aluMemCycles(3, Rcycles, 0);
	return 1;
}

static int OP_LDRD_STRD_OFFSET_PRE_INDEX(const u32 i)
{
	DEBUG_DYNARC(ldrd_strd_pre)
	u8 Rd_num = REG_POS(i, 12);
	
	if (Rd_num == 14)
	{
		printf("OP_LDRD_STRD_OFFSET_PRE_INDEX: use R14!!!!\n");
		return 0; // TODO: exception
	}
	if (Rd_num & 0x1)
	{
		printf("OP_LDRD_STRD_OFFSET_PRE_INDEX: ERROR!!!!\n");
		return 0; // TODO: exception
	}

	auto addr = regman.alloc_temp32();
	c.MOV(addr, map_reg(16));

	// I bit - immediate or register
	if (BIT22(i))
	{
		IMM_OFF;
		BIT23(i)?c.ADD(addr, addr, rhs):c.SUB(addr, addr, rhs);
	}
	else
		BIT23(i)?c.ADD(addr, addr, map_reg(0)):c.SUB(addr, addr, map_reg(0));

	if (BIT5(i))		// Store
	{
		c.MOV(W0, addr);
		regman.call(X2, op_strd_tab[PROCNUM][Rd_num], true);
		c.MOV(Rcycles, W0);

		if (BIT21(i)) // W bit - writeback
			c.MOV(map_reg(16), addr);
		emit_MMU_aluMemCycles(3, Rcycles, 0);
	}
	else				// Load
	{
		if (BIT21(i)) // W bit - writeback
			c.MOV(map_reg(16), addr);
		
		c.MOV(W0, addr);
		regman.call(X2, op_ldrd_tab[PROCNUM][Rd_num], true);
		c.MOV(Rcycles, W0);

		emit_MMU_aluMemCycles(3, Rcycles, 0);
	}
	return 1;
}

//-----------------------------------------------------------------------------
//   SWP/SWPB
//-----------------------------------------------------------------------------
template<int PROCNUM>
static u32 FASTCALL op_swp(u32 adr, u32 *Rd, u32 Rs)
{
	u32 tmp = ROR(READ32(cpu->mem_if->data, adr), (adr & 3)<<3);
	WRITE32(cpu->mem_if->data, adr, Rs);
	*Rd = tmp;
	return (MMU_memAccessCycles<PROCNUM,32,MMU_AD_READ>(adr) + MMU_memAccessCycles<PROCNUM,32,MMU_AD_WRITE>(adr));
}
template<int PROCNUM>
static u32 FASTCALL op_swpb(u32 adr, u32 *Rd, u32 Rs)
{
	u32 tmp = READ8(cpu->mem_if->data, adr);
	WRITE8(cpu->mem_if->data, adr, Rs);
	*Rd = tmp;
	return (MMU_memAccessCycles<PROCNUM,8,MMU_AD_READ>(adr) + MMU_memAccessCycles<PROCNUM,8,MMU_AD_WRITE>(adr));
}

typedef u32 FASTCALL (*OP_SWP_SWPB)(u32, u32*, u32);
static const OP_SWP_SWPB op_swp_tab[2][2] = {{ op_swp<0>, op_swp<1> }, { op_swpb<0>, op_swpb<1> }};

static int op_swp_(const u32 i, int b)
{
	DEBUG_DYNARC(swp)
	auto func_tmp = regman.alloc_temp64();

	c.MOV(W0, map_reg(16));
	c.ADD(X1, RCPU, offsetof(armcpu_t, R) + 4*REG_POS(i,12));
	c.UBFX(W2, map_reg(0), 0, b?8:32);

	regman.call(func_tmp, op_swp_tab[b][PROCNUM], true);
	c.MOV(Rcycles, W0);
	c.LDR(INDEX_UNSIGNED, map_reg(12), RCPU, offsetof(armcpu_t, R) + 4*REG_POS(i,12));

	regman.free_temp64(func_tmp);

	emit_MMU_aluMemCycles(4, Rcycles, 0);
	return 1;
}

static int OP_SWP(const u32 i) { return op_swp_(i, 0); }
static int OP_SWPB(const u32 i) { return op_swp_(i, 1); }

//-----------------------------------------------------------------------------
//   LDMIA / LDMIB / LDMDA / LDMDB / STMIA / STMIB / STMDA / STMDB
//-----------------------------------------------------------------------------
static u32 popregcount(u32 x)
{
	uint32_t pop = 0;
	for(; x; x>>=1)
		pop += x&1;
	return pop;
}

static u64 get_reg_list(u32 reg_mask, int dir)
{
	u64 regs = 0;
	for(int j=0; j<16; j++)
	{
		int k = dir<0 ? j : 15-j;
		if(BIT_N(reg_mask,k))
			regs = (regs << 4) | k;
	}
	return regs;
}

#ifdef ASMJIT_X64
// generic needs to spill regs and main doesn't; if it's inlined gcc isn't smart enough to keep the spills out of the common case.
#define LDM_INLINE NOINLINE
#else
// spills either way, and we might as well save codesize by not having separate functions
#define LDM_INLINE INLINE
#endif

template <int PROCNUM, bool store, int dir>
static LDM_INLINE FASTCALL u32 OP_LDM_STM_generic(u32 adr, u64 regs, int n)
{
	u32 cycles = 0;
	adr &= ~3;
	do {
		if(store) _MMU_write32<PROCNUM>(adr, cpu->R[regs&0xF]);
		else cpu->R[regs&0xF] = _MMU_read32<PROCNUM>(adr);
		cycles += MMU_memAccessCycles<PROCNUM,32,store?MMU_AD_WRITE:MMU_AD_READ>(adr);
		adr += 4*dir;
		regs >>= 4;
	} while(--n > 0);
	return cycles;
}

#ifdef ENABLE_ADVANCED_TIMING
#define ADV_CYCLES cycles += MMU_memAccessCycles<PROCNUM,32,store?MMU_AD_WRITE:MMU_AD_READ>(adr);
#else
#define ADV_CYCLES
#endif

template <int PROCNUM, bool store, int dir>
static LDM_INLINE FASTCALL u32 OP_LDM_STM_other(u32 adr, u64 regs, int n)
{
	u32 cycles = 0;
	adr &= ~3;
#ifndef ENABLE_ADVANCED_TIMING
	cycles = n * MMU_memAccessCycles<PROCNUM,32,store?MMU_AD_WRITE:MMU_AD_READ>(adr);
#endif
	do {
		if(PROCNUM==ARMCPU_ARM9)
			if(store) _MMU_ARM9_write32(adr, cpu->R[regs&0xF]);
			else cpu->R[regs&0xF] = _MMU_ARM9_read32(adr);
		else
			if(store) _MMU_ARM7_write32(adr, cpu->R[regs&0xF]);
			else cpu->R[regs&0xF] = _MMU_ARM7_read32(adr);
		ADV_CYCLES;
		adr += 4*dir;
		regs >>= 4;
	} while(--n > 0);
	return cycles;
}

template <int PROCNUM, bool store, int dir, bool null_compiled>
static FORCEINLINE FASTCALL u32 OP_LDM_STM_main(u32 adr, u64 regs, int n, u8 *ptr, u32 cycles)
{
#ifdef ENABLE_ADVANCED_TIMING
	cycles = 0;
#endif
	uintptr_t *func = (uintptr_t *)&JIT_COMPILED_FUNC(adr, PROCNUM);

#define OP(j) { \
	/* no need to zero functions in DTCM, since we can't execute from it */ \
	if(null_compiled && store) \
	{ \
		*func = 0; \
		*(func+1) = 0; \
	} \
	int Rd = ((uintptr_t)regs >> (j*4)) & 0xF; \
	if(store) *(u32*)ptr = cpu->R[Rd]; \
	else cpu->R[Rd] = *(u32*)ptr; \
	ADV_CYCLES; \
	func += 2*dir; \
	adr += 4*dir; \
	ptr += 4*dir; }

	do {
		OP(0);
		if(n == 1) break;
		OP(1);
		if(n == 2) break;
		OP(2);
		if(n == 3) break;
		OP(3);
		regs >>= 16;
		n -= 4;
	} while(n > 0);
	return cycles;
#undef OP
#undef ADV_CYCLES
}

template <int PROCNUM, bool store, int dir>
static u32 FASTCALL OP_LDM_STM(u32 adr, u64 regs, int n)
{
	// TODO use classify_adr?
	u32 cycles;
	u8 *ptr;

	if((adr ^ (adr + (dir>0 ? (n-1)*4 : -15*4))) & ~0x3FFF) // a little conservative, but we don't want to run too many comparisons
	{
		// the memory region spans a page boundary, so we can't factor the address translation out of the loop
		return OP_LDM_STM_generic<PROCNUM, store, dir>(adr, regs, n);
	}
	else if(PROCNUM==ARMCPU_ARM9 && (adr & ~0x3FFF) == MMU.DTCMRegion)
	{
		// don't special-case DTCM cycles, even though that would be both faster and more accurate,
		// because that wouldn't match the non-jitted version with !ACCOUNT_FOR_DATA_TCM_SPEED
		ptr = MMU.ARM9_DTCM + (adr & 0x3FFC);
		cycles = n * MMU_memAccessCycles<PROCNUM,32,store?MMU_AD_WRITE:MMU_AD_READ>(adr);
		if(store)
			return OP_LDM_STM_main<PROCNUM, store, dir, 0>(adr, regs, n, ptr, cycles);
	}
	else if((adr & 0x0F000000) == 0x02000000)
	{
		ptr = MMU.MAIN_MEM + (adr & _MMU_MAIN_MEM_MASK32);
		cycles = n * ((PROCNUM==ARMCPU_ARM9) ? 4 : 2);
	}
	else if(PROCNUM==ARMCPU_ARM7 && !store && (adr & 0xFF800000) == 0x03800000)
	{
		ptr = MMU.ARM7_ERAM + (adr & 0xFFFC);
		cycles = n;
	}
	else if(PROCNUM==ARMCPU_ARM7 && !store && (adr & 0xFF800000) == 0x03000000)
	{
		ptr = MMU.SWIRAM + (adr & 0x7FFC);
		cycles = n;
	}
	else
		return OP_LDM_STM_other<PROCNUM, store, dir>(adr, regs, n);

	return OP_LDM_STM_main<PROCNUM, store, dir, store>(adr, regs, n, ptr, cycles);
}

typedef u32 FASTCALL (*LDMOpFunc)(u32,u64,int);
static const LDMOpFunc op_ldm_stm_tab[2][2][2] = {{
	{ OP_LDM_STM<0,0,-1>, OP_LDM_STM<0,0,+1> },
	{ OP_LDM_STM<0,1,-1>, OP_LDM_STM<0,1,+1> },
},{
	{ OP_LDM_STM<1,0,-1>, OP_LDM_STM<1,0,+1> },
	{ OP_LDM_STM<1,1,-1>, OP_LDM_STM<1,1,+1> },
}};

static void call_ldm_stm(ARM64Reg adr, u32 bitmask, bool store, int dir)
{
	if(bitmask)
	{
		auto func_temp = regman.alloc_temp64();
		
		c.MOV(W0, adr);
		c.MOVI2R(W1, get_reg_list(bitmask, dir));
		c.MOVI2R(W2, popregcount(bitmask));
		regman.call(func_temp, op_ldm_stm_tab[PROCNUM][store][dir>0], true);
		c.MOV(Rcycles, W0);

		regman.free_temp64(func_temp);
	}
	else
		bb_constant_cycles++;
}

static int op_bx(ARM64Reg srcreg, bool blx, bool test_thumb);
static int op_bx_thumb(ARM64Reg srcreg, bool blx, bool test_thumb);

static int op_ldm_stm(u32 i, bool store, int dir, bool before, bool writeback)
{
	DEBUG_DYNARC(ldm_stm)
	u32 bitmask = i & 0xFFFF;
	u32 pop = popregcount(bitmask);

	auto adr = regman.alloc_temp32();
	if(before) {
		if (dir > 0) c.ADD(adr, adr, 4);
		else if (dir < 0) c.SUB(adr, adr, 4);
	} else
		c.MOV(adr, map_reg(16));

	call_ldm_stm(adr, bitmask, store, dir);

	if(BIT15(i) && !store)
	{
		op_bx(regman.map_reg32(15), 0, PROCNUM == ARMCPU_ARM9);
	}

	if(writeback)
	{
		if(store || !(i & (1 << REG_POS(i,16))))
		{
			JIT_COMMENT("--- writeback");
			if (dir > 0)
				c.ADD(map_reg(16), map_reg(16), 4*pop);
			else if (dir < 0)
				c.SUB(map_reg(16), map_reg(16), 4*pop);
		}
		else 
		{
			u32 bitlist = (~((2 << REG_POS(i,16))-1)) & 0xFFFF;
			if(i & bitlist)
			{
				JIT_COMMENT("--- writeback");
				if (dir > 0)
					c.ADD(map_reg(16), adr, 4*(pop-before));
				else if (dir < 0)
					c.SUB(map_reg(16), adr, 4*(pop-before));
			}
		}
	}

	emit_MMU_aluMemCycles(store ? 1 : 2, Rcycles, pop);
	return 1;
}

static int OP_LDMIA(const u32 i) { return op_ldm_stm(i, 0, +1, 0, 0); }
static int OP_LDMIB(const u32 i) { return op_ldm_stm(i, 0, +1, 1, 0); }
static int OP_LDMDA(const u32 i) { return op_ldm_stm(i, 0, -1, 0, 0); }
static int OP_LDMDB(const u32 i) { return op_ldm_stm(i, 0, -1, 1, 0); }
static int OP_LDMIA_W(const u32 i) { return op_ldm_stm(i, 0, +1, 0, 1); }
static int OP_LDMIB_W(const u32 i) { return op_ldm_stm(i, 0, +1, 1, 1); }
static int OP_LDMDA_W(const u32 i) { return op_ldm_stm(i, 0, -1, 0, 1); }
static int OP_LDMDB_W(const u32 i) { return op_ldm_stm(i, 0, -1, 1, 1); }

static int OP_STMIA(const u32 i) { return op_ldm_stm(i, 1, +1, 0, 0); }
static int OP_STMIB(const u32 i) { return op_ldm_stm(i, 1, +1, 1, 0); }
static int OP_STMDA(const u32 i) { return op_ldm_stm(i, 1, -1, 0, 0); }
static int OP_STMDB(const u32 i) { return op_ldm_stm(i, 1, -1, 1, 0); }
static int OP_STMIA_W(const u32 i) { return op_ldm_stm(i, 1, +1, 0, 1); }
static int OP_STMIB_W(const u32 i) { return op_ldm_stm(i, 1, +1, 1, 1); }
static int OP_STMDA_W(const u32 i) { return op_ldm_stm(i, 1, -1, 0, 1); }
static int OP_STMDB_W(const u32 i) { return op_ldm_stm(i, 1, -1, 1, 1); }

static int op_ldm_stm2(u32 i, bool store, int dir, bool before, bool writeback)
{
	DEBUG_DYNARC(ldm_stm2)

	u32 bitmask = i & 0xFFFF;
	u32 pop = popregcount(bitmask);
	bool bit15 = BIT15(i);

	//printf("ARM%c: %s R%d:%08X, bitmask %02X\n", PROCNUM?'7':'9', (store?"STM":"LDM"), REG_POS(i, 16), cpu->R[REG_POS(i, 16)], bitmask);
	u32 adr_first = cpu->R[REG_POS(i, 16)];

	auto adr = regman.alloc_temp32();
	if(before)
		if (dir > 0) c.ADD(adr, map_reg(16), 4);
		else if(dir < 0) c.SUB(adr, map_reg(16), 4);
	else
		c.MOV(adr, map_reg(16));
	
	ARM64Reg oldmode;

	if (!bit15 || store)
	{
		//if((cpu->CPSR.bits.mode==USR)||(cpu->CPSR.bits.mode==SYS)) { printf("ERROR1\n"); return 1; }
		//oldmode = armcpu_switchMode(cpu, SYS);
		c.MOV(X0, RCPU);
		c.MOVI2R(W1, SYS);
		regman.call(X2, armcpu_switchMode, true);
		oldmode = regman.alloc_temp32();
		c.MOV(oldmode, W0);
	}

	call_ldm_stm(adr, bitmask, store, dir);

	if(!bit15 || store)
	{
		c.MOV(X0, RCPU);
		c.MOV(W1, oldmode);
		regman.call(X2, armcpu_switchMode, true);

		regman.free_temp32(oldmode);
	}
	else
	{
		S_DST_R15;
	}

	if(writeback)
	{
		// FIXME und zwar wirklich
		if(store || !(i & (1 << REG_POS(i,16))))
		{
			JIT_COMMENT("--- writeback");
			if (dir > 0)
				c.ADD(map_reg(16), map_reg(16), 4*pop);
			else if (dir < 0)
				c.SUB(map_reg(16), map_reg(16), 4*pop);
		}
		else 
		{
			u32 bitlist = (~((2 << REG_POS(i,16))-1)) & 0xFFFF;
			if(i & bitlist)
			{
				JIT_COMMENT("--- writeback");
				if (dir > 0)
					c.ADD(map_reg(16), adr, 4*(pop-before));
				else if (dir < 0)
					c.SUB(map_reg(16), adr, 4*(pop-before));
			}
		}
	}

	emit_MMU_aluMemCycles(store ? 1 : 2, Rcycles, pop);
	return 1;
}

static int OP_LDMIA2(const u32 i) { return op_ldm_stm2(i, 0, +1, 0, 0); }
static int OP_LDMIB2(const u32 i) { return op_ldm_stm2(i, 0, +1, 1, 0); }
static int OP_LDMDA2(const u32 i) { return op_ldm_stm2(i, 0, -1, 0, 0); }
static int OP_LDMDB2(const u32 i) { return op_ldm_stm2(i, 0, -1, 1, 0); }
static int OP_LDMIA2_W(const u32 i) { return op_ldm_stm2(i, 0, +1, 0, 1); }
static int OP_LDMIB2_W(const u32 i) { return op_ldm_stm2(i, 0, +1, 1, 1); }
static int OP_LDMDA2_W(const u32 i) { return op_ldm_stm2(i, 0, -1, 0, 1); }
static int OP_LDMDB2_W(const u32 i) { return op_ldm_stm2(i, 0, -1, 1, 1); }

static int OP_STMIA2(const u32 i) { return op_ldm_stm2(i, 1, +1, 0, 0); }
static int OP_STMIB2(const u32 i) { return op_ldm_stm2(i, 1, +1, 1, 0); }
static int OP_STMDA2(const u32 i) { return op_ldm_stm2(i, 1, -1, 0, 0); }
static int OP_STMDB2(const u32 i) { return op_ldm_stm2(i, 1, -1, 1, 0); }
static int OP_STMIA2_W(const u32 i) { return op_ldm_stm2(i, 1, +1, 0, 1); }
static int OP_STMIB2_W(const u32 i) { return op_ldm_stm2(i, 1, +1, 1, 1); }
static int OP_STMDA2_W(const u32 i) { return op_ldm_stm2(i, 1, -1, 0, 1); }
static int OP_STMDB2_W(const u32 i) { return op_ldm_stm2(i, 1, -1, 1, 1); }

//-----------------------------------------------------------------------------
//   Branch
//-----------------------------------------------------------------------------
#define SIGNEXTEND_11(i) (((s32)i<<21)>>21)
#define SIGNEXTEND_24(i) (((s32)i<<8)>>8)

static int op_b(u32 i, bool bl)
{
	DEBUG_DYNARC(b)
	u32 dst = bb_r15 + (SIGNEXTEND_24(i) << 2);
	if(CONDITION(i)==0xF)
	{
		if(bl)
			dst += 2;
		c.ORRI2R(RCPSR, RCPSR, 1 << 5);
	}
	if(bl || CONDITION(i)==0xF)
		c.MOVI2R(regman.map_reg32(14), bb_next_instruction);

	auto tmp = regman.alloc_temp32();
	c.MOVI2R(tmp, dst);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
	regman.free_temp32(tmp);
	return 1;
}

static int OP_B(const u32 i) { return op_b(i, 0); }
static int OP_BL(const u32 i) { return op_b(i, 1); }

static int op_bx(ARM64Reg srcreg, bool blx, bool test_thumb)
{
	DEBUG_DYNARC(bx)
	auto dst = regman.alloc_temp32();
	c.MOV(dst, srcreg);

	if(test_thumb)
	{
		auto tmp = regman.alloc_temp32();
		c.UBFX(tmp, dst, 0, 1);
		c.BFI(RCPSR, tmp, 5, 1);
		c.LSL(tmp, tmp, 1);
		c.ORRI2R(tmp, tmp, 0xFFFFFFFC);
		c.AND(dst, dst, tmp);
		regman.free_temp32(tmp);
	}
	else
		c.ANDI2R(dst, dst, 0xFFFFFFFC);

	if(blx)
		c.MOVI2R(regman.map_reg32(14), bb_next_instruction);
	c.STR(INDEX_UNSIGNED, dst, RCPU, offsetof(armcpu_t, instruct_adr));

	regman.free_temp32(dst);
	return 1;
}

static int OP_BX(const u32 i) { return op_bx(map_reg(0), 0, 1); }
static int OP_BLX_REG(const u32 i) { return op_bx(map_reg(0), 1, 1); }

//-----------------------------------------------------------------------------
//   CLZ
//-----------------------------------------------------------------------------
static int OP_CLZ(const u32 i)
{
	DEBUG_DYNARC(clz)
	c.CLZ(map_reg(12), map_reg(0));
	return 1;
}

//-----------------------------------------------------------------------------
//   MCR / MRC
//-----------------------------------------------------------------------------

// precalculate region masks/sets from cp15 register ----- JIT
// TODO: rewrite to asm
static void maskPrecalc(u32 _num)
{
#define precalc(num) {  \
	u32 mask = 0, set = 0xFFFFFFFF ; /* (x & 0) == 0xFF..FF is allways false (disabled) */  \
	if (BIT_N(cp15.protectBaseSize[num],0)) /* if region is enabled */ \
	{    /* reason for this define: naming includes var */  \
		mask = CP15_MASKFROMREG(cp15.protectBaseSize[num]) ;   \
		set = CP15_SETFROMREG(cp15.protectBaseSize[num]) ; \
		if (CP15_SIZEIDENTIFIER(cp15.protectBaseSize[num])==0x1F)  \
		{   /* for the 4GB region, u32 suffers wraparound */   \
			mask = 0 ; set = 0 ;   /* (x & 0) == 0  is allways true (enabled) */  \
		} \
	}  \
	cp15.setSingleRegionAccess(num, mask, set) ;  \
}
	switch(_num)
	{
		case 0: precalc(0); break;
		case 1: precalc(1); break;
		case 2: precalc(2); break;
		case 3: precalc(3); break;
		case 4: precalc(4); break;
		case 5: precalc(5); break;
		case 6: precalc(6); break;
		case 7: precalc(7); break;

		case 0xFF:
			precalc(0);
			precalc(1);
			precalc(2);
			precalc(3);
			precalc(4);
			precalc(5);
			precalc(6);
			precalc(7);
		break;
	}
#undef precalc
}

#define _maskPrecalc(num) \
{ \
	c.MOVI2R(W0, num); \
	regman.call(X2, maskPrecalc, false); \
}

static int OP_MCR(const u32 i)
{
	DEBUG_DYNARC(mrc)
	if (PROCNUM == ARMCPU_ARM7) return 0;

	u32 cpnum = REG_POS(i, 8);
	if(cpnum != 15)
	{
		// TODO - exception?
		printf("JIT: MCR P%i, 0, R%i, C%i, C%i, %i, %i (don't allocated coprocessor)\n", 
			cpnum, REG_POS(i, 12), REG_POS(i, 16), REG_POS(i, 0), (i>>21)&0x7, (i>>5)&0x7);
		return 2;
	}
	if (REG_POS(i, 12) == 15)
	{
		printf("JIT: MCR Rd=R15\n");
		return 2;
	}

	u8 CRn =  REG_POS(i, 16);		// Cn
	u8 CRm =  REG_POS(i, 0);		// Cm
	u8 opcode1 = ((i>>21)&0x7);		// opcode1
	u8 opcode2 = ((i>>5)&0x7);		// opcode2

	auto bb_cp15 = regman.alloc_temp64();
	c.MOVP2R(bb_cp15, &cp15);
	auto data = regman.alloc_temp32();
	c.MOV(data, map_reg(12));

	bool bUnknown = false;
	switch(CRn)
	{
		case 1:
			if((opcode1==0) && (opcode2==0) && (CRm==0))
			{
				auto tmp = regman.alloc_temp32();
				// On the NDS bit0,2,7,12..19 are R/W, Bit3..6 are always set, all other bits are always zero.
				//MMU.ARM9_RW_MODE = BIT7(val);
				auto bb_mmu = regman.alloc_temp64();
				c.MOVP2R(bb_mmu, &MMU);
				c.UBFX(tmp, data, 7, 1);
				c.STRB(INDEX_UNSIGNED, tmp, bb_mmu, offsetof(MMU_struct, ARM9_RW_MODE));
				regman.free_temp64(bb_mmu);
				//cpu->intVector = 0xFFFF0000 * (BIT13(val));
				c.UBFX(tmp, data, 13, 1);
				c.TST(tmp, tmp);
				c.MOVI2R(tmp, 0xFFFF0000);
				c.CSEL(tmp, tmp, WZR, CC_NEQ);
				c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, intVector));
				//cpu->LDTBit = !BIT15(val); //TBit
				c.UBFX(tmp, data, 15, 1);
				c.ORN(tmp, WZR, tmp);
				c.STRB(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, LDTBit));
				//ctrl = (val & 0x000FF085) | 0x00000078;
				c.ANDI2R(data, data, 0x000FF085, tmp);
				c.ORRI2R(data, data, 0x00000078, tmp);
				c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, ctrl));
				regman.free_temp32(tmp);
				break;
			}
			bUnknown = true;
			break;
		case 2:
			if((opcode1==0) && (CRm==0))
			{
				switch(opcode2)
				{
					case 0:
						// DCConfig = val;
						c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DCConfig));
						break;
					case 1:
						// ICConfig = val;
						c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, ICConfig));
						break;
					default:
						bUnknown = true;
						break;
				}
				break;
			}
			bUnknown = true;
			break;
		case 3:
			if((opcode1==0) && (opcode2==0) && (CRm==0))
			{
				//writeBuffCtrl = val;
				c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, writeBuffCtrl));
				break;
			}
			bUnknown = true;
			break;
		case 5:
			if((opcode1==0) && (CRm==0))
			{
				switch(opcode2)
				{
					case 2:
						//DaccessPerm = val;
						c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DaccessPerm));
						_maskPrecalc(0xFF);
						break;
					case 3:
						//IaccessPerm = val;
						c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, IaccessPerm));
						_maskPrecalc(0xFF);
						break;
					default:
						bUnknown = true;
						break;
				}
			}
			bUnknown = true;
			break;
		case 6:
			if((opcode1==0) && (opcode2==0))
			{
				if (CRm < 8)
				{
					//protectBaseSize[CRm] = val;
					c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, protectBaseSize) + (CRm * sizeof(u32)));
					_maskPrecalc(CRm);
					break;
				}
			}
			bUnknown = true;
			break;
		case 7:
			if((CRm==0)&&(opcode1==0)&&((opcode2==4)))
			{
				//CP15wait4IRQ;
				auto tmp = regman.alloc_temp32();
				c.MOVI2R(tmp, CPU_FREEZE_IRQ_IE_IF);
				c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, freeze));
				regman.free_temp32(tmp);
				//IME set deliberately omitted: only SWI sets IME to 1
				break;
			}
			bUnknown = true;
			break;
		case 9:
			if((opcode1==0))
			{
				switch(CRm)
				{
					case 0:
						switch(opcode2)
						{
							case 0:
								//DcacheLock = val;
								c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DcacheLock));
								break;
							case 1:
								//IcacheLock = val;
								c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, IcacheLock));
								break;
							default:
								bUnknown = true;
								break;
						}
					case 1:
						switch(opcode2)
						{
							case 0:
								{
									//MMU.DTCMRegion = DTCMRegion = val & 0x0FFFF000;
									c.ANDI2R(data, data, 0x0FFFF000);
									auto bb_mmu = regman.alloc_temp64();
									c.MOVP2R(bb_mmu, &MMU);
									c.STR(INDEX_UNSIGNED, data, bb_mmu, offsetof(MMU_struct, DTCMRegion));
									c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DTCMRegion));
									regman.free_temp64(bb_mmu);
								}
								break;
							case 1:
								{
									//ITCMRegion = val;
									//ITCM base is not writeable!
									auto bb_mmu = regman.alloc_temp64();
									c.MOVP2R(bb_mmu, &MMU);
									c.STR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, ITCMRegion));
									c.MOVZ(data, 0);
									c.STR(INDEX_UNSIGNED, data, bb_mmu, offsetof(MMU_struct, ITCMRegion));
									regman.free_temp64(bb_mmu);
								}
								break;
							default:
								bUnknown = true;
								break;
						}
				}
				break;
			}
			bUnknown = true;
			break;
		case 13:
			bUnknown = true;
			break;
		case 15:
			bUnknown = true;
			break;
		default:
			bUnknown = true;
			break;
	}

	regman.free_temp32(data);
	regman.free_temp64(bb_cp15);

	if (bUnknown)
	{
		//printf("Unknown MCR command: MRC P15, 0, R%i, C%i, C%i, %i, %i\n", REG_POS(i, 12), CRn, CRm, opcode1, opcode2);
		return 1;
	}

	return 1;
}
static int OP_MRC(const u32 i)
{
	DEBUG_DYNARC(mrc)
	if (PROCNUM == ARMCPU_ARM7) return 0;

	u32 cpnum = REG_POS(i, 8);
	if(cpnum != 15)
	{
		printf("MRC P%i, 0, R%i, C%i, C%i, %i, %i (don't allocated coprocessor)\n", cpnum, REG_POS(i, 12), REG_POS(i, 16), REG_POS(i, 0), (i>>21)&0x7, (i>>5)&0x7);
		return 2;
	}

	u8 CRn =  REG_POS(i, 16);		// Cn
	u8 CRm =  REG_POS(i, 0);		// Cm
	u8 opcode1 = ((i>>21)&0x7);		// opcode1
	u8 opcode2 = ((i>>5)&0x7);		// opcode2

	auto bb_cp15 = regman.alloc_temp64();
	c.MOVP2R(bb_cp15, &cp15);
	auto data = regman.alloc_temp32();
	
	bool bUnknown = false;
	switch(CRn)
	{
		case 0:
			if((opcode1 == 0)&&(CRm==0))
			{
				switch(opcode2)
				{
					case 1:
						// *R = cacheType;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, cacheType));
						break;
					case 2:
						// *R = TCMSize;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, TCMSize));
						break;
					default:		// FIXME
						// *R = IDCode;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, IDCode));
						break;
				}
				break;
			}
			bUnknown = true;
			break;
		
		case 1:
			if((opcode1==0) && (opcode2==0) && (CRm==0))
			{
				// *R = ctrl;
				c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, ctrl));
				break;
			}
			bUnknown = true;
			break;
		
		case 2:
			if((opcode1==0) && (CRm==0))
			{
				switch(opcode2)
				{
					case 0:
						// *R = DCConfig;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DCConfig));
						break;
					case 1:
						// *R = ICConfig;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, ICConfig));
						break;
					default:
						bUnknown = true;
						break;
				}
				break;
			}
			bUnknown = true;
			break;
			
		case 3:
			if((opcode1==0) && (opcode2==0) && (CRm==0))
			{
				// *R = writeBuffCtrl;
				c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, writeBuffCtrl));
				break;
			}
			bUnknown = true;
			break;
			
		case 5:
			if((opcode1==0) && (CRm==0))
			{
				switch(opcode2)
				{
					case 2:
						// *R = DaccessPerm;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DaccessPerm));
						break;
					case 3:
						// *R = IaccessPerm;
						c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, IaccessPerm));
						break;
					default:
						bUnknown = true;
						break;
				}
				break;
			}
			bUnknown = true;
			break;
			
		case 6:
			if((opcode1==0) && (opcode2==0))
			{
				if (CRm < 8)
				{
					// *R = protectBaseSize[CRm];
					c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, protectBaseSize) + CRm * sizeof(u32));
					break;
				}
			}
			bUnknown = true;
			break;
			
		case 7:
			bUnknown = true;
			break;

		case 9:
			if(opcode1 == 0)
			{
				switch(CRm)
				{
					case 0:
						switch(opcode2)
						{
							case 0:
								//*R = DcacheLock;
								c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DcacheLock));
								break;
							case 1:
								//*R = IcacheLock;
								c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, IcacheLock));
								break;
							default:
								bUnknown = true;
								break;
						}

					case 1:
						switch(opcode2)
						{
							case 0:
								//*R = DTCMRegion;
								c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, DTCMRegion));
								break;
							case 1:
								//*R = ITCMRegion;
								c.LDR(INDEX_UNSIGNED, data, bb_cp15, offsetof(armcp15_t, ITCMRegion));
								break;
							default:
								bUnknown = true;
								break;
						}
				}
				//
				break;
			}
			bUnknown = true;
			break;
			
		case 13:
			bUnknown = true;
			break;
			
		case 15:
			bUnknown = true;
			break;

		default:
			bUnknown = true;
			break;
	}

	if (bUnknown)
	{
		//printf("Unknown MRC command: MRC P15, 0, R%i, C%i, C%i, %i, %i\n", REG_POS(i, 12), CRn, CRm, opcode1, opcode2);
		return 1;
	}

	if (REG_POS(i, 12) == 15)	// set NZCV
	{
		//CPSR.bits.N = BIT31(data);
		//CPSR.bits.Z = BIT30(data);
		//CPSR.bits.C = BIT29(data);
		//CPSR.bits.V = BIT28(data);
		c.ANDI2R(data, data, 0xF0000000);
		c.ANDI2R(RCPSR, RCPSR, 0x0FFFFFFF);
		c.ORR(RCPSR, RCPSR, data);
	}
	else
		c.MOV(map_reg(12), data);

	regman.free_temp32(data);
	regman.free_temp64(bb_cp15);

	return 1;
}

u32 op_swi(u8 swinum)
{
	DEBUG_DYNARC(swi)
	if(cpu->swi_tab)
	{
		// ?
		regman.call(X2, ARM_swi_tab[PROCNUM][swinum], true);
		c.ADD(Rcycles, W0, 3);
		return 1;
	}

	auto oldCPSR = regman.alloc_temp32();
	c.MOV(oldCPSR, RCPSR);
	c.MOV(X0, RCPU);
	c.MOVI2R(W1, SVC);
	regman.call(X2, armcpu_switchMode, true);
	c.MOVI2R(regman.map_reg32(14), bb_next_instruction);
	c.STR(INDEX_UNSIGNED, oldCPSR, RCPU, offsetof(armcpu_t, SPSR.val));
	c.ANDI2R(RCPSR, RCPSR, ~(1 << 5));
	c.ORRI2R(RCPSR, RCPSR, 1 << 7);
	
	c.MOVI2R(oldCPSR, cpu->intVector+0x08); // recycle
	c.STR(INDEX_UNSIGNED, oldCPSR, RCPU, offsetof(armcpu_t, next_instruction));
	
	return 1;
}

static int OP_SWI(const u32 i) { return op_swi((i >> 16) & 0x1F); }

//-----------------------------------------------------------------------------
//   BKPT
//-----------------------------------------------------------------------------
static int OP_BKPT(const u32 i) { printf("JIT: unimplemented OP_BKPT\n"); return 0; }

//-----------------------------------------------------------------------------
//   THUMB
//-----------------------------------------------------------------------------
#define OP_SHIFTS_IMM(a64inst, rshift) \
	DEBUG_DYNARC(thumb_shift_imm) \
	auto rcf = regman.alloc_temp32(); \
	u8 cf_change = 1; \
	const u32 rhs = ((i>>6) & 0x1F); \
	c.a64inst(map_reg_thumb(0), map_reg_thumb(3), rhs); \
	c.TST(map_reg_thumb(0), map_reg_thumb(0)); \
	if (rshift) \
		c.BFI(rcf, map_reg_thumb(0), rhs - 1, 1); \
	else \
		c.BFI(rcf, map_reg_thumb(0), 32-rhs, 1); \
	SET_NZC; \
	return 1;

#define OP_SHIFTS_REG(a64inst, lshift, sign) \
	DEBUG_DYNARC(thumb_shift_reg) \
	bool rhs_is_imm = false; \
	u8 cf_change = 1; \
	auto rhs = regman.alloc_temp32(); \
	auto rcf = regman.alloc_temp32(); \
	auto tmp = regman.alloc_temp32(); \
	c.ANDI2R(tmp, map_reg_thumb(3), 0xff); \
	c.a64inst(rhs, map_reg_thumb(0), tmp); \
	c.CMP(map_reg_thumb(3), 32); \
	if (!sign) \
	{ \
		c.MOVZ(tmp, 0); \
		c.CSEL(rhs, rhs, tmp, CC_LT); \
	} \
	else \
	{ \
		c.MOV(tmp, map_reg_thumb(0), ArithOption(tmp, ST_ASR, 31)); \
		c.CSEL(rhs, rhs, tmp, CC_LT); \
	} \
	auto __no_shift = c.CBZ(map_reg_thumb(3)); \
	c.UBFX(rcf, RCPSR, 29, 1); \
	/* rcf = rm >> (32 - rn) */ \
	if (lshift) \
	{ \
		c.MOVZ(tmp, 32); \
		c.SUB(tmp, tmp, map_reg_thumb(3)); \
		c.LSRV(rcf, map_reg_thumb(0), tmp); \
	} \
	else \
	/* rcf = rm >> (rn - 1) */ \
	{ \
		c.SUB(tmp, map_reg_thumb(3), 1); \
		c.LSRV(rcf, map_reg_thumb(0), tmp); \
	} \
	/*c.ANDI2R(rcf, rcf, 1);*/ \
	if (sign) \
		c.UBFX(tmp, map_reg_thumb(0), 31, 1); \
	else \
		c.MOVZ(tmp, 0); \
	c.CSEL(rcf, tmp, rcf, sign ? CC_GE : CC_GT); \
	c.SetJumpTarget(__no_shift); \
	c.MOV(map_reg_thumb(0), rhs); \
	regman.free_temp32(tmp); \
	regman.free_temp32(rhs); \
	regman.free_temp32(tmp); \
	SET_NZC \
	return 1;

#define OP_LOGIC(a64Inst) \
	DEBUG_DYNARC(thumb_logic) \
	c.a64Inst(map_reg_thumb(0), map_reg_thumb(0), map_reg_thumb(3)); \
	c.TST(map_reg_thumb(0), map_reg_thumb(0)); \
	SET_NZ(0); \
	return 1;

//-----------------------------------------------------------------------------
//   LSL / LSR / ASR / ROR
//-----------------------------------------------------------------------------
static int OP_LSL_0(const u32 i)
{
	DEBUG_DYNARC(thumb_shift_0)
	c.MOV(map_reg_thumb(0), map_reg_thumb(3));
	c.TST(map_reg_thumb(0), map_reg_thumb(0));
	SET_NZ(0);
	return 1;
}
static int OP_LSL(const u32 i) { OP_SHIFTS_IMM(LSL, 0); }
static int OP_LSL_REG(const u32 i) { OP_SHIFTS_REG(LSLV, 1, 0); }
static int OP_LSR_0(const u32 i) 
{
	DEBUG_DYNARC(thumb_shift_0)
	bool cf_change = true;
	auto rcf = regman.alloc_temp32();
	c.UBFX(rcf, map_reg_thumb(0), 31, 1);
	c.MOVZ(map_reg_thumb(0), 0);
	c.TST(map_reg_thumb(0), map_reg_thumb(0));
	SET_NZC;
	return 1;
}
static int OP_LSR(const u32 i) { OP_SHIFTS_IMM(LSR, 1); }
static int OP_LSR_REG(const u32 i) { OP_SHIFTS_REG(LSRV, 0, 0); }
static int OP_ASR_0(const u32 i)
{
	DEBUG_DYNARC(thumb_shift_0)
	u8 cf_change = 1;
	auto rcf = regman.alloc_temp32();
	c.ASR(map_reg_thumb(0), map_reg_thumb(3), 31);
	c.UBFX(rcf, map_reg_thumb(0), 31, 1);
	SET_NZC;
	return 1;
}
static int OP_ASR(const u32 i) { OP_SHIFTS_IMM(ASR, 1); }
static int OP_ASR_REG(const u32 i) { OP_SHIFTS_REG(ASRV, 0, 1); }

//-----------------------------------------------------------------------------
//   ROR
//-----------------------------------------------------------------------------
static int OP_ROR_REG(const u32 i)
{
	DEBUG_DYNARC(thumb_ror)
	bool rhs_is_imm = false; \
	bool cf_change = 1; \
	auto rhs = regman.alloc_temp32(); \
	auto tmp = regman.alloc_temp32(); \
	auto rcf = regman.alloc_temp32(); \
	c.ANDI2R(tmp, map_reg_thumb(3), 0xff); \
	c.RORV(rhs, map_reg_thumb(0), tmp); \
	c.UBFX(rcf, RCPSR, 29, 1); \
	auto __zero = c.CBZ(map_reg_thumb(3)); \
	c.SUB(tmp, map_reg_thumb(3), 1); \
	c.LSRV(rcf, map_reg_thumb(0), tmp); /* LSRV wraps automatically around */ \
	c.SetJumpTarget(__zero); \
	regman.free_temp32(tmp);
	c.TST(rhs, rhs);
	c.MOV(map_reg_thumb(0), rhs);
	SET_NZC;
	regman.free_temp32(rhs);

	return 1;
}

//-----------------------------------------------------------------------------
//   AND / ORR / EOR / BIC
//-----------------------------------------------------------------------------
static int OP_AND(const u32 i) { OP_LOGIC(AND); }
static int OP_ORR(const u32 i) { OP_LOGIC(ORR); }
static int OP_EOR(const u32 i) { OP_LOGIC(EOR); }
static int OP_BIC(const u32 i) { OP_LOGIC(BIC); }

//-----------------------------------------------------------------------------
//   NEG
//-----------------------------------------------------------------------------
static int OP_NEG(const u32 i)
{
	DEBUG_DYNARC(thumb_neg)
	c.NEG(map_reg_thumb(0), map_reg_thumb(3));
	c.TST(map_reg_thumb(0), map_reg_thumb(0));
	SET_NZCV(1);
	return 1;
}

//-----------------------------------------------------------------------------
//   ADD
//-----------------------------------------------------------------------------
static int OP_ADD_IMM3(const u32 i) 
{
	DEBUG_DYNARC(thumb_add)
	u32 imm3 = (i >> 6) & 0x07;

	c.ADDS(map_reg_thumb(0), map_reg_thumb(3), imm3);
	SET_NZCV(0);
	return 1;
}
static int OP_ADD_IMM8(const u32 i) 
{
	DEBUG_DYNARC(thumb_add)
	c.ADDS(map_reg_thumb(8), map_reg_thumb(8), i & 0xff);
	SET_NZCV(0);

	return 1; 
}
static int OP_ADD_REG(const u32 i) 
{
	DEBUG_DYNARC(thumb_add)
	//cpu->R[REG_NUM(i, 0)] = cpu->R[REG_NUM(i, 3)] + cpu->R[REG_NUM(i, 6)];
	c.ADDS(map_reg_thumb(0), map_reg_thumb(3), map_reg_thumb(6));
	SET_NZCV(0);
	return 1; 
}
static int OP_ADD_SPE(const u32 i)
{
	DEBUG_DYNARC(thumb_add)
	u32 Rd = _REG_NUM(i, 0) | ((i>>4)&8);
	//cpu->R[Rd] += cpu->R[REG_POS(i, 3)];

	c.ADD(regman.map_reg32(Rd), regman.map_reg32(Rd), map_reg(3));
	
	if(Rd==15)
		c.STR(INDEX_UNSIGNED, regman.map_reg32(15), RCPU, offsetof(armcpu_t, next_instruction));
		
	return 1;
}

static int OP_ADD_2PC(const u32 i)
{
	DEBUG_DYNARC(thumb_add)
	u32 imm = ((i&0xFF)<<2);
	c.MOVI2R(map_reg_thumb(8), (bb_r15 & 0xFFFFFFFC) + imm);
	return 1;
}

static int OP_ADD_2SP(const u32 i)
{
	DEBUG_DYNARC(thumb_add)
	u32 imm = ((i&0xFF)<<2);
	//cpu->R[REG_NUM(i, 8)] = cpu->R[13] + ((i&0xFF)<<2);
	c.ADD(map_reg_thumb(8), regman.map_reg32(13), ((i&0xFF)<<2));
	return 1;
}

//-----------------------------------------------------------------------------
//   SUB
//-----------------------------------------------------------------------------
static int OP_SUB_IMM3(const u32 i)
{
	DEBUG_DYNARC(thumb_sub)
	u32 imm3 = (i >> 6) & 0x07;

	c.SUBS(map_reg_thumb(0), map_reg_thumb(3), imm3);
	SET_NZCV(0);
	return 1;
}
static int OP_SUB_IMM8(const u32 i)
{
	DEBUG_DYNARC(thumb_sub)
	//cpu->R[REG_NUM(i, 8)] -= imm8;
	c.SUBS(map_reg_thumb(8), map_reg_thumb(8), (i & 0xFF));
	SET_NZCV(0);
	return 1; 
}
static int OP_SUB_REG(const u32 i)
{
	DEBUG_DYNARC(thumb_sub)
	// cpu->R[REG_NUM(i, 0)] = cpu->R[REG_NUM(i, 3)] - cpu->R[REG_NUM(i, 6)];
	c.SUBS(map_reg_thumb(0), map_reg_thumb(3), map_reg_thumb(6));
	SET_NZCV(0);
	return 1; 
}

//-----------------------------------------------------------------------------
//   ADC
//-----------------------------------------------------------------------------
static int OP_ADC_REG(const u32 i)
{
	DEBUG_DYNARC(thumb_adc_sbc)
	GET_CARRY(0);
	c.ADCS(map_reg_thumb(0), map_reg_thumb(0), map_reg_thumb(3));
	SET_NZCV(0);
	return 1;
}

//-----------------------------------------------------------------------------
//   SBC
//-----------------------------------------------------------------------------
static int OP_SBC_REG(const u32 i)
{
	DEBUG_DYNARC(thumb_adc_sbc)
	GET_CARRY(0);
	c.SBCS(map_reg_thumb(0), map_reg_thumb(0), map_reg_thumb(3));
	SET_NZCV(0);
	return 1;
}

//-----------------------------------------------------------------------------
//   MOV / MVN
//-----------------------------------------------------------------------------
static int OP_MOV_IMM8(const u32 i)
{
	DEBUG_DYNARC(thumb_mov)
	c.MOVZ(map_reg_thumb(8), i & 0xff);
	c.TST(map_reg_thumb(8), map_reg_thumb(8));
	SET_NZ(0);
	return 1;
}

static int OP_MOV_SPE(const u32 i)
{
	DEBUG_DYNARC(thumb_mov)
	u32 Rd = _REG_NUM(i, 0) | ((i>>4)&8);
	//cpu->R[Rd] = cpu->R[REG_POS(i, 3)];
	c.MOV(regman.map_reg32(Rd), map_reg(3));
	if(Rd == 15)
	{
		c.STR(INDEX_UNSIGNED, regman.map_reg32(15), RCPU, offsetof(armcpu_t, next_instruction));
		bb_constant_cycles += 2;
	}
	
	return 1;
}

static int OP_MVN(const u32 i)
{
	DEBUG_DYNARC(thumb_mov)
	c.MVN(map_reg_thumb(0), map_reg_thumb(3));
	c.TST(map_reg_thumb(0), map_reg_thumb(0));
	SET_NZ(0);
	return 1;
}

//-----------------------------------------------------------------------------
//   MUL
//-----------------------------------------------------------------------------
static int OP_MUL_REG(const u32 i) 
{
	DEBUG_DYNARC(thumb_mul)
	auto tmp = regman.alloc_temp32();
	c.MOV(tmp, map_reg_thumb(0));
	c.MUL(map_reg_thumb(0), map_reg_thumb(0), map_reg_thumb(3));
	c.TST(map_reg_thumb(0), map_reg_thumb(0));
	
	SET_NZ(0);
	if (PROCNUM == ARMCPU_ARM7)
		c.MOVZ(Rcycles, 4);
	else
		MUL_Mxx_END(tmp, 0, 1);
	
	regman.free_temp32(tmp);
	return 1;
}

//-----------------------------------------------------------------------------
//   CMP / CMN
//-----------------------------------------------------------------------------
static int OP_CMP_IMM8(const u32 i) 
{
	DEBUG_DYNARC(thumb_cmp_tst)
	c.CMPI2R(map_reg_thumb(8), i & 0xFF);
	SET_NZCV(1);
	return 1; 
}

static int OP_CMP(const u32 i) 
{
	DEBUG_DYNARC(thumb_cmp_tst)
	c.CMP(map_reg_thumb(0), map_reg_thumb(3));
	SET_NZCV(1);
	return 1; 
}

static int OP_CMP_SPE(const u32 i) 
{
	DEBUG_DYNARC(thumb_cmp_tst)
	u32 Rn = (i&7) | ((i>>4)&8);
	c.CMP(regman.map_reg32(Rn), map_reg(3));
	SET_NZCV(1);
	return 1; 
}

static int OP_CMN(const u32 i) 
{
	DEBUG_DYNARC(thumb_cmp_tst)
	c.CMN(map_reg_thumb(0), map_reg_thumb(3));
	SET_NZCV(0);
	return 1; 
}

//-----------------------------------------------------------------------------
//   TST
//-----------------------------------------------------------------------------
static int OP_TST(const u32 i)
{
	DEBUG_DYNARC(thumb_cmp_tst)
	c.TST(map_reg_thumb(0), map_reg_thumb(3));
	SET_NZ(0);
	return 1;
}

//-----------------------------------------------------------------------------
//   STR / LDR / STRB / LDRB
//-----------------------------------------------------------------------------
#define STR_THUMB(mem_op, offset) \
	DEBUG_DYNARC(thumb_str) \
	auto addr = W0; \
	auto data = W1; \
	u32 adr_first = cpu->R[_REG_NUM(i, 3)]; \
	 \
	c.MOV(addr, map_reg_thumb(3)); \
	if ((offset) != -1) \
	{ \
		if ((offset) != 0) \
		{ \
			c.ADD(addr, addr, (u32)offset); \
			adr_first += (u32)(offset); \
		} \
	} \
	else \
	{ \
		c.ADD(addr, addr, map_reg_thumb(6)); \
		adr_first += cpu->R[_REG_NUM(i, 6)]; \
	} \
	c.MOV(data, map_reg_thumb(0)); \
	regman.call(X2, mem_op##_tab[PROCNUM][classify_adr(adr_first,1)], true); \
	c.MOV(Rcycles, W0); \
	return 1;

#define LDR_THUMB(mem_op, offset) \
	DEBUG_DYNARC(thumb_ldr) \
	auto addr = W0; \
	auto data = X1; \
	u32 adr_first = cpu->R[_REG_NUM(i, 3)]; \
	 \
	c.MOV(addr, map_reg_thumb(3)); \
	if ((offset) != -1) \
	{ \
		if ((offset) != 0) \
		{ \
			c.ADD(addr, addr, (u32)(offset)); \
			adr_first += (u32)(offset); \
		} \
	} \
	else \
	{ \
		c.ADD(addr, addr, map_reg_thumb(6)); \
		adr_first += cpu->R[_REG_NUM(i, 6)]; \
	} \
	c.ADD(data, RCPU, 4*(i&0x7)); \
	regman.call(X2, mem_op##_tab[PROCNUM][classify_adr(adr_first,0)], true); \
	c.MOV(Rcycles, W0); \
	return 1;

static int OP_STRB_IMM_OFF(const u32 i) { STR_THUMB(STRB, ((i>>6)&0x1F)); }
static int OP_LDRB_IMM_OFF(const u32 i) { LDR_THUMB(LDRB, ((i>>6)&0x1F)); }
static int OP_STRB_REG_OFF(const u32 i) { STR_THUMB(STRB, -1); } 
static int OP_LDRB_REG_OFF(const u32 i) { LDR_THUMB(LDRB, -1); }
static int OP_LDRSB_REG_OFF(const u32 i) { LDR_THUMB(LDRSB, -1); }

static int OP_STRH_IMM_OFF(const u32 i) { STR_THUMB(STRH, ((i>>5)&0x3E)); }
static int OP_LDRH_IMM_OFF(const u32 i) { LDR_THUMB(LDRH, ((i>>5)&0x3E)); }
static int OP_STRH_REG_OFF(const u32 i) { STR_THUMB(STRH, -1); }
static int OP_LDRH_REG_OFF(const u32 i) { LDR_THUMB(LDRH, -1); }
static int OP_LDRSH_REG_OFF(const u32 i) { LDR_THUMB(LDRSH, -1); } 

static int OP_STR_IMM_OFF(const u32 i) { STR_THUMB(STR, ((i>>4)&0x7C)); }
static int OP_LDR_IMM_OFF(const u32 i) { LDR_THUMB(LDR, ((i>>4)&0x7C)); } // FIXME: tempValue = (tempValue>>adr) | (tempValue<<(32-adr));
static int OP_STR_REG_OFF(const u32 i) { STR_THUMB(STR, -1); }
static int OP_LDR_REG_OFF(const u32 i) { LDR_THUMB(LDR, -1); }

static int OP_STR_SPREL(const u32 i)
{
	DEBUG_DYNARC(thumb_str)
	u32 imm = ((i&0xFF)<<2);
	u32 adr_first = cpu->R[13] + imm;

	auto addr = W0;
	if (imm) c.ADD(addr, regman.map_reg32(13), imm);
	else c.MOV(addr, regman.map_reg32(13));
	auto data = regman.alloc_temp32();
	c.MOV(data, W1);
	regman.call(X2, STR_tab[PROCNUM][classify_adr(adr_first,1)], true);
	c.MOV(Rcycles, W0);
	return 1;
}

static int OP_LDR_SPREL(const u32 i)
{
	DEBUG_DYNARC(thumb_ldr)
	u32 imm = ((i&0xFF)<<2);
	u32 adr_first = cpu->R[13] + imm;
	int dstReg = (i>>8)&0x7;
	
	auto addr = W0;
	if (imm) c.ADD(addr, regman.map_reg32(13), imm);
	else c.MOV(addr, regman.map_reg32(13));
	auto data = X1;
	c.ADD(data, RCPU, dstReg*4);
	regman.call(X2, LDR_tab[PROCNUM][classify_adr(adr_first,0)], true);
	c.MOV(Rcycles, W0);
	c.LDR(INDEX_UNSIGNED, regman.map_reg32(dstReg), RCPU, offsetof(armcpu_t, R)+dstReg*4);
	return 1;
}

static int OP_LDR_PCREL(const u32 i)
{
	DEBUG_DYNARC(thumb_ldr)
	u32 imm = ((i&0xFF)<<2);
	u32 adr_first = (bb_r15 & 0xFFFFFFFC) + imm;
	auto addr = W0;
	auto data = X1;
	int dstReg = (i>>8)&0x7;
	c.MOVI2R(addr, adr_first);
	c.ADD(data, RCPU, dstReg*4);
	regman.call(X2, LDR_tab[PROCNUM][classify_adr(adr_first,0)], true);
	c.MOV(Rcycles, W0);
	c.LDR(INDEX_UNSIGNED, regman.map_reg32(dstReg), RCPU, offsetof(armcpu_t, R)+dstReg*4);
	return 1;
}

//-----------------------------------------------------------------------------
//   STMIA / LDMIA
//-----------------------------------------------------------------------------
static int op_ldm_stm_thumb(u32 i, bool store)
{
	DEBUG_DYNARC(thumb_ldm_stm)
	u32 bitmask = i & 0xFF;
	u32 pop = popregcount(bitmask);

	//if (BIT_N(i, _REG_NUM(i, 8)))
	//	printf("WARNING - %sIA with Rb in Rlist (THUMB)\n", store?"STM":"LDM");

	auto adr = regman.alloc_temp32();
	c.MOV(adr, map_reg_thumb(8));

	call_ldm_stm(adr, bitmask, store, 1);

	// ARM_REF:	THUMB: Causes base register write-back, and is not optional
	// ARM_REF:	If the base register <Rn> is specified in <registers>, the final value of <Rn> is the loaded value
	//			(not the written-back value).
	if (store)
		c.ADD(map_reg_thumb(8), map_reg_thumb(8), 4*pop);
	else
	{
		if (!BIT_N(i, _REG_NUM(i, 8)))
			c.ADD(map_reg_thumb(8), map_reg_thumb(8), 4*pop);
	}
	regman.free_temp32(adr);

	emit_MMU_aluMemCycles(store ? 2 : 3, Rcycles, pop);
	return 1;
}

static int OP_LDMIA_THUMB(const u32 i) { return op_ldm_stm_thumb(i, 0); }
static int OP_STMIA_THUMB(const u32 i) { return op_ldm_stm_thumb(i, 1); }

//-----------------------------------------------------------------------------
//   Adjust SP
//-----------------------------------------------------------------------------
static int OP_ADJUST_P_SP(const u32 i) { 
	DEBUG_DYNARC(thumb_adjust_sp)
	c.ADD(regman.map_reg32(13), regman.map_reg32(13), ((i&0x7F)<<2)); 
	return 1; 
}
static int OP_ADJUST_M_SP(const u32 i) {
	DEBUG_DYNARC(thumb_adjust_sp)
	c.SUB(regman.map_reg32(13), regman.map_reg32(13), ((i&0x7F)<<2)); 
	return 1; 
}

//-----------------------------------------------------------------------------
//   PUSH / POP
//-----------------------------------------------------------------------------
static int op_push_pop(u32 i, bool store, bool pc_lr)
{
	DEBUG_DYNARC(thumb_push_pop)
	u32 bitmask = (i & 0xFF);
	bitmask |= pc_lr << (store ? 14 : 15);
	u32 pop = popregcount(bitmask);
	int dir = store ? -1 : 1;

	auto adr = regman.alloc_temp32();
	if(store) c.SUB(adr, regman.map_reg32(13), 4);
	else c.MOV(adr, regman.map_reg32(13));

	call_ldm_stm(adr, bitmask, store, dir);

	if(pc_lr && !store)
		op_bx_thumb(regman.map_reg32(15), 0, PROCNUM == ARMCPU_ARM9);
	
	if (dir > 0) c.ADD(regman.map_reg32(13), regman.map_reg32(13), 4*pop);
	else if (dir < 0) c.SUB(regman.map_reg32(13), regman.map_reg32(13), 4*pop);
	
	emit_MMU_aluMemCycles(store ? (pc_lr?4:3) : (pc_lr?5:2), Rcycles, pop);
	return 1;
}

static int OP_PUSH(const u32 i)    { return op_push_pop(i, 1, 0); }
static int OP_PUSH_LR(const u32 i) { return op_push_pop(i, 1, 1); }
static int OP_POP(const u32 i)     { return op_push_pop(i, 0, 0); }
static int OP_POP_PC(const u32 i)  { return op_push_pop(i, 0, 1); }

//-----------------------------------------------------------------------------
//   Branch
//-----------------------------------------------------------------------------
static int OP_B_COND(const u32 i)
{
	DEBUG_DYNARC(thumb_branch)
	u32 dst = bb_r15 + ((u32)((s8)(i&0xFF))<<1);

	auto tmp = regman.alloc_temp32();
	c.MOVI2R(tmp, bb_next_instruction);
	FixupBranch branch;
	bool branches = emit_branch((i>>8)&0xF, branch);
	c.MOVI2R(tmp, dst);
	c.ADD(Rtotal_cycles, Rtotal_cycles, 2);
	if (branches) c.SetJumpTarget(branch);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
	regman.free_temp32(tmp);

	return 1;
}

static int OP_B_UNCOND(const u32 i)
{
	DEBUG_DYNARC(thumb_branch)
	u32 dst = bb_r15 + (SIGNEXTEND_11(i)<<1);
	auto tmp = regman.alloc_temp32();
	c.MOVI2R(tmp, dst);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
	regman.free_temp32(tmp);
	return 1;
}

static int OP_BLX(const u32 i)
{
	DEBUG_DYNARC(thumb_branch)
	auto dst = regman.alloc_temp32();
	c.ADD(dst, regman.map_reg32(14), (i&0x7FF) << 1);
	c.ANDI2R(dst, dst, 0xFFFFFFFC);
	c.STR(INDEX_UNSIGNED, dst, RCPU, offsetof(armcpu_t, instruct_adr));
	c.MOVI2R(regman.map_reg32(14), bb_next_instruction | 1);
	// reset T bit
	c.ANDI2R(RCPSR, RCPSR, ~(1<<5));
	return 1;
}

static int OP_BL_10(const u32 i)
{
	DEBUG_DYNARC(thumb_branch)
	u32 dst = bb_r15 + (SIGNEXTEND_11(i)<<12);
	c.MOVI2R(regman.map_reg32(14), dst);
	return 1;
}

static int OP_BL_11(const u32 i) 
{
	DEBUG_DYNARC(thumb_branch)
	auto dst = regman.alloc_temp32();
	c.ADD(dst, regman.map_reg32(14), (i&0x7FF) << 1);
	c.STR(INDEX_UNSIGNED, dst, RCPU, offsetof(armcpu_t, instruct_adr));
	c.MOVI2R(regman.map_reg32(14), bb_next_instruction | 1);
	return 1;
}

static int op_bx_thumb(ARM64Reg srcreg, bool blx, bool test_thumb)
{
	DEBUG_DYNARC(thumb_branch)
	auto dst = regman.alloc_temp32();
	auto thumb = regman.alloc_temp32();

	c.MOV(dst, srcreg);
	c.MOV(thumb, dst);
	c.ANDI2R(thumb, thumb, 1);
	if (blx)
		c.MOVI2R(regman.map_reg32(14), bb_next_instruction|1);
	if(test_thumb)
	{
		auto mask = regman.alloc_temp32();
		c.MOVI2R(mask, 0xFFFFFFFC);
		c.ORR(mask, mask, thumb, ArithOption(mask, ST_LSL, 1));
		c.AND(dst, dst, mask);
		regman.free_temp32(mask);
	}
	else
		c.ANDI2R(dst, dst, 0xFFFFFFFE);

	c.ANDI2R(RCPSR, RCPSR, ~(1<< 5));
	c.BFI(RCPSR, thumb, 5, 1);

	c.STR(INDEX_UNSIGNED, dst, RCPU, offsetof(armcpu_t, instruct_adr));

	regman.free_temp32(dst);
	regman.free_temp32(thumb);
	return 1;
}

static int op_bx_thumbR15()
{
	DEBUG_DYNARC(thumb_branch)
	const u32 r15 = (bb_r15 & 0xFFFFFFFC);
	auto tmp = regman.alloc_temp32();
	c.MOVI2R(tmp, r15);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
	c.MOV(regman.map_reg32(15), tmp);
	c.ANDI2R(RCPSR, RCPSR, (u32)~(1<< 5));

	return 1;
}

static int OP_BX_THUMB(const u32 i) { if (REG_POS(i, 3) == 15) return op_bx_thumbR15(); return op_bx_thumb(map_reg_thumb(3), 0, 0); }
static int OP_BLX_THUMB(const u32 i) { return op_bx_thumb(map_reg_thumb(3), 1, 1); }

static int OP_SWI_THUMB(const u32 i) { return op_swi(i & 0x1F); }

//-----------------------------------------------------------------------------
//   Unimplemented; fall back to the C versions
//-----------------------------------------------------------------------------

#define OP_UND           NULL
#define OP_LDREX         NULL
#define OP_STREX         NULL
#define OP_LDC_P_IMM_OFF NULL
#define OP_LDC_M_IMM_OFF NULL
#define OP_LDC_P_PREIND  NULL
#define OP_LDC_M_PREIND  NULL
#define OP_LDC_P_POSTIND NULL
#define OP_LDC_M_POSTIND NULL
#define OP_LDC_OPTION    NULL
#define OP_STC_P_IMM_OFF NULL
#define OP_STC_M_IMM_OFF NULL
#define OP_STC_P_PREIND  NULL
#define OP_STC_M_PREIND  NULL
#define OP_STC_P_POSTIND NULL
#define OP_STC_M_POSTIND NULL
#define OP_STC_OPTION    NULL
#define OP_CDP           NULL

#define OP_UND_THUMB     NULL
#define OP_BKPT_THUMB    NULL

//-----------------------------------------------------------------------------
//   Dispatch table
//-----------------------------------------------------------------------------

typedef int (*ArmOpCompiler)(u32);
static const ArmOpCompiler arm_instruction_compilers[4096] = {
#define TABDECL(x) x
#include "instruction_tabdef.inc"
#undef TABDECL
};

static const ArmOpCompiler thumb_instruction_compilers[1024] = {
#define TABDECL(x) x
#include "thumb_tabdef.inc"
#undef TABDECL
};

//-----------------------------------------------------------------------------
//   Generic instruction wrapper
//-----------------------------------------------------------------------------

template<int PROCNUM, int thumb>
static u32 FASTCALL OP_DECODE()
{
	u32 cycles;
	u32 adr = cpu->instruct_adr;
	if(thumb)
	{
		cpu->next_instruction = adr + 2;
		cpu->R[15] = adr + 4;
		u32 opcode = _MMU_read16<PROCNUM, MMU_AT_CODE>(adr);
		_armlog(PROCNUM, adr, opcode);
		cycles = thumb_instructions_set[PROCNUM][opcode>>6](opcode);
	}
	else
	{
		cpu->next_instruction = adr + 4;
		cpu->R[15] = adr + 8;
		u32 opcode = _MMU_read32<PROCNUM, MMU_AT_CODE>(adr);
		_armlog(PROCNUM, adr, opcode);
		if(CONDITION(opcode) == 0xE || TEST_COND(CONDITION(opcode), CODE(opcode), cpu->CPSR))
			cycles = arm_instructions_set[PROCNUM][INSTRUCTION_INDEX(opcode)](opcode);
		else
			cycles = 1;
	}
	cpu->instruct_adr = cpu->next_instruction;
	return cycles;
}

static const ArmOpCompiled op_decode[2][2] = { OP_DECODE<0,0>, OP_DECODE<0,1>, OP_DECODE<1,0>, OP_DECODE<1,1> };

//-----------------------------------------------------------------------------
//   Compiler
//-----------------------------------------------------------------------------

static u32 instr_attributes(u32 opcode)
{
	return bb_thumb ? thumb_attributes[opcode>>6]
		 : instruction_attributes[INSTRUCTION_INDEX(opcode)];
}

static bool instr_is_branch(u32 opcode)
{
	u32 x = instr_attributes(opcode);
	
	if(bb_thumb)
	{
		// merge OP_BL_10+OP_BL_11
		if (x & MERGE_NEXT) return false;
		return (x & BRANCH_ALWAYS)
		    || ((x & BRANCH_POS0) && ((opcode&7) | ((opcode>>4)&8)) == 15)
			|| (x & BRANCH_SWI)
		    || (x & JIT_BYPASS);
	}
	else
		return (x & BRANCH_ALWAYS)
		    || ((x & BRANCH_POS12) && REG_POS(opcode,12) == 15)
		    || ((x & BRANCH_LDM) && BIT15(opcode))
			|| (x & BRANCH_SWI)
		    || (x & JIT_BYPASS);
}

static bool instr_uses_r15(u32 opcode)
{
	u32 x = instr_attributes(opcode);
	if(bb_thumb)
		return ((x & SRCREG_POS0) && ((opcode&7) | ((opcode>>4)&8)) == 15)
			|| ((x & SRCREG_POS3) && REG_POS(opcode,3) == 15)
			|| (x & JIT_BYPASS);
	else
		return ((x & SRCREG_POS0) && REG_POS(opcode,0) == 15)
		    || ((x & SRCREG_POS8) && REG_POS(opcode,8) == 15)
		    || ((x & SRCREG_POS12) && REG_POS(opcode,12) == 15)
		    || ((x & SRCREG_POS16) && REG_POS(opcode,16) == 15)
		    || ((x & SRCREG_STM) && BIT15(opcode))
		    || (x & JIT_BYPASS);
}

static bool instr_is_conditional(u32 opcode)
{
	if(bb_thumb) return false;
	
	return !(CONDITION(opcode) == 0xE
	         || (CONDITION(opcode) == 0xF && CODE(opcode) == 5));
}

static int instr_cycles(u32 opcode)
{
	u32 x = instr_attributes(opcode);
	u32 c = (x & INSTR_CYCLES_MASK);
	if(c == INSTR_CYCLES_VARIABLE)
	{
		if ((x & BRANCH_SWI) && !cpu->swi_tab)
			return 3;
		
		return 0;
	}
	if(instr_is_branch(opcode) && !(instr_attributes(opcode) & (BRANCH_ALWAYS|BRANCH_LDM)))
		c += 2;
	return c;
}

static bool instr_does_prefetch(u32 opcode)
{
	u32 x = instr_attributes(opcode);
	if(bb_thumb)
		return thumb_instruction_compilers[opcode>>6]
			   && (x & BRANCH_ALWAYS);
	else
		return instr_is_branch(opcode) && arm_instruction_compilers[INSTRUCTION_INDEX(opcode)]
			   && ((x & BRANCH_ALWAYS) || (x & BRANCH_LDM));
}

static const char *disassemble(u32 opcode)
{
	if(bb_thumb)
		return thumb_instruction_names[opcode>>6];
	static char str[100];
	strcpy(str, arm_instruction_names[INSTRUCTION_INDEX(opcode)]);
	static const char *conds[16] = {"EQ","NE","CS","CC","MI","PL","VS","VC","HI","LS","GE","LT","GT","LE","AL","NV"};
	if(instr_is_conditional(opcode))
	{
		strcat(str, ".");
		strcat(str, conds[CONDITION(opcode)]);
	}
	return str;
}
static void printout(u32 a, u32 b) { printf("value %d %d\n", a, b); }

static void sync_r15(u32 opcode, bool is_last, bool force)
{
	if(instr_does_prefetch(opcode))
	{
		if(force)
		{
			//assert(!instr_uses_r15(opcode));
			JIT_COMMENT("sync_r15: force instruct_adr %08Xh (PREFETCH)", bb_adr);
			auto tmp = regman.alloc_temp32();
			c.MOVI2R(tmp, bb_next_instruction);
			c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
			c.MOVI2R(W0, 2);
			c.MOV(W1, tmp);
			/*regman.prepare_helper_call();
			c.QuickCallFunction(X2, printout);
			regman.post_helper_call();*/
			regman.free_temp32(tmp);
		}
	}
	else
	{
		if(force || (instr_attributes(opcode) & JIT_BYPASS) || (instr_attributes(opcode) & BRANCH_SWI) || (is_last && !instr_is_branch(opcode)))
		{
			
			JIT_COMMENT("sync_r15: next_instruction %08Xh - %s%s%s%s", bb_next_instruction,
				force?" FORCE":"",
				(instr_attributes(opcode) & JIT_BYPASS)?" BYPASS":"",
				(instr_attributes(opcode) & BRANCH_SWI)?" SWI":"",
				(is_last && !instr_is_branch(opcode))?" LAST":""
			);
			auto tmp = regman.alloc_temp32();
			c.MOVI2R(tmp, bb_next_instruction);
			c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction));
			c.MOVI2R(W0, 3);
			c.MOV(W1, tmp);
			/*regman.prepare_helper_call();
			c.QuickCallFunction(X2, printout);
			regman.post_helper_call();*/
			regman.free_temp32(tmp);
		}
		if(instr_uses_r15(opcode))
		{
			JIT_COMMENT("sync_r15: R15 %08Xh (USES R15)", bb_r15);
			c.MOVI2R(regman.map_reg32(15), bb_r15);
		}
		if(instr_attributes(opcode) & JIT_BYPASS)
		{
			JIT_COMMENT("sync_r15: instruct_adr %08Xh (JIT_BYPASS)", bb_adr);
			auto tmp = regman.alloc_temp32();
			c.MOVI2R(tmp, bb_adr);
			c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
			c.MOVI2R(W0, 32);
			c.MOV(W1, tmp);
			/*regman.prepare_helper_call();
			c.QuickCallFunction(X2, printout);
			regman.post_helper_call();*/
			regman.free_temp32(tmp);
		}
	}
}

static void test() { printf("actually calling something\n"); }

static void print_out(u32 cycA, u32 cycB)
{
	printf("cycles %d %d\n", cycA, cycB);
}

static bool emit_branch(int cond, FixupBranch& to)
{
	if (cond != 0xe)
	{
		regman.load_cpsr();
		c._MSR(FIELD_NZCV, (ARM64Reg)(X0 + reg32_get_n(RCPSR)));
		/*regman.prepare_helper_call();
		c.QuickCallFunction(X2, test);
		regman.post_helper_call();*/
		to = c.B((CCFlags)(cond ^ 1));
		return true;
	}
	return false;
}

static void emit_armop_call(u32 opcode)
{
	ArmOpCompiler fc = bb_thumb?	thumb_instruction_compilers[opcode>>6]:
									arm_instruction_compilers[INSTRUCTION_INDEX(opcode)];
	if (!instr_uses_r15(opcode) && !instr_is_branch(opcode) && fc && fc(opcode)){
		printf("not interpreted %s\n", disassemble(opcode));
		regman.save_cpsr();
		return;
	}
	
	//printf("calling interpreter\n");
	JIT_COMMENT("call interpreter");
	c.MOVI2R(W0, opcode);
	OpFunc f = bb_thumb ? thumb_instructions_set[PROCNUM][opcode>>6]
	                     : arm_instructions_set[PROCNUM][INSTRUCTION_INDEX(opcode)];
	regman.call(X2, f, true);
	c.MOV(Rcycles, W0);

	regman.load_cpsr();
}

static void _armlog(u8 proc, u32 addr, u32 opcode)
{
#if 0
#if 0
	fprintf(stderr, "\t\t;R0:%08X R1:%08X R2:%08X R3:%08X R4:%08X R5:%08X R6:%08X R7:%08X R8:%08X R9:%08X\n\t\t;R10:%08X R11:%08X R12:%08X R13:%08X R14:%08X R15:%08X| next %08X, N:%i Z:%i C:%i V:%i\n",
		cpu->R[0],  cpu->R[1],  cpu->R[2],  cpu->R[3],  cpu->R[4],  cpu->R[5],  cpu->R[6],  cpu->R[7], 
		cpu->R[8],  cpu->R[9],  cpu->R[10],  cpu->R[11],  cpu->R[12],  cpu->R[13],  cpu->R[14],  cpu->R[15],
		cpu->next_instruction, cpu->CPSR.bits.N, cpu->CPSR.bits.Z, cpu->CPSR.bits.C, cpu->CPSR.bits.V);
#endif
	#define INDEX22(i) ((((i)>>16)&0xFF0)|(((i)>>4)&0xF))
	char dasmbuf[4096];
	if(cpu->CPSR.bits.T)
		des_thumb_instructions_set[((opcode)>>6)&1023](addr, opcode, dasmbuf);
	else
		des_arm_instructions_set[INDEX22(opcode)](addr, opcode, dasmbuf);
	#undef INDEX22
	fprintf(stderr, "%s%c %08X\t%08X \t%s\n", cpu->CPSR.bits.T?"THUMB":"ARM", proc?'7':'9', addr, opcode, dasmbuf); 
#else
	return;
#endif
}

template<int PROCNUM>
static u32 compile_basicblock()
{
#if LOG_JIT
	bool has_variable_cycles = FALSE;
#endif
	u32 interpreted_cycles = 0;
	u32 start_adr = cpu->instruct_adr;
	u32 opcode = 0;
	
	bb_thumb = cpu->CPSR.bits.T;
	bb_opcodesize = bb_thumb ? 2 : 4;

	if (!JIT_MAPPED(start_adr & 0x0FFFFFFF, PROCNUM))
	{
		printf("JIT: use unmapped memory address %08X\n", start_adr);
		execute = false;
		return 1;
	}

#if LOG_JIT
	fprintf(stderr, "adr %08Xh %s%c\n", start_adr, ARMPROC.CPSR.bits.T ? "THUMB":"ARM", PROCNUM?'7':'9');
#endif

	libnx::jitTransitionToWritable(&jit_page);
	auto f = (ArmOpCompiled)((u32*)c.GetCodePtr() - jit_rw_addr + jit_rx_addr);

	BitSet32 stashed_regs({
		reg64_get_n(RCPU), reg32_get_n(Rtotal_cycles), reg32_get_n(RCPSR), 
		reg32_get_n(Rcycles), 30,
		19, 20, 21, 22, 23, 24});
	c.ABI_PushRegisters(stashed_regs);
	c.MOVZ(Rtotal_cycles, 0);
	c.MOVP2R(RCPU, &ARMPROC);

	regman.reset();
	regman.load_cpsr();

#if (PROFILER_JIT_LEVEL > 0)
	JIT_COMMENT("Profiler ptr");
	bb_profiler = c.newGpVar(kX86VarTypeGpz);
	c.mov(bb_profiler, (uintptr_t)&profiler_counter[PROCNUM]);
#endif

	// we fetch the opcodes in advance, so we'll able to better predict which registers to load/unload
	u32 opcodes[CommonSettings.jit_max_block_size];
	u32 opcodes_count = 0;
	while(opcodes_count < CommonSettings.jit_max_block_size)
	{
		u32 adr = start_adr + opcodes_count * bb_opcodesize;
		opcodes[opcodes_count] = bb_thumb ?
			_MMU_read16<PROCNUM, MMU_AT_CODE>(adr)
			: _MMU_read32<PROCNUM, MMU_AT_CODE>(adr);

		opcodes_count++;

		if(instr_is_branch(opcodes[opcodes_count-1])) break;
	}

	bb_constant_cycles = 0;
	for(u32 i=0; i<opcodes_count; i++)
	{
		bb_adr = start_adr + (i * bb_opcodesize);
		opcode = opcodes[i];

#if LOG_JIT
		char dasmbuf[1024] = {0};
		if(bb_thumb)
			des_thumb_instructions_set[opcode>>6](bb_adr, opcode, dasmbuf);
		else
			des_arm_instructions_set[INSTRUCTION_INDEX(opcode)](bb_adr, opcode, dasmbuf);
		fprintf(stderr, "%08X\t%s\t\t; %s \n", bb_adr, dasmbuf, disassemble(opcode));
#endif

		u32 cycles = instr_cycles(opcode);
		
#if LOG_JIT
		if (instr_is_conditional(opcode) && (cycles > 1) || (cycles == 0))
			has_variable_cycles = TRUE;
#endif
		bb_constant_cycles += instr_is_conditional(opcode) ? 1 : cycles;

		//printf("%s (PC:%08X)\n", disassemble(opcode), bb_adr);
		JIT_COMMENT("%s (PC:%08X)", disassemble(opcode), bb_adr);

#if (PROFILER_JIT_LEVEL > 0)
		JIT_COMMENT("*** profiler - counter");
		if (bb_thumb)
			c.add(profiler_counter_thumb(opcode), 1);
		else
			c.add(profiler_counter_arm(opcode), 1);
#endif

		printf("%s\n", disassemble(opcode));
		// das Problem ist:
		// letzte Instruktion ist Branch, next_instruction ist aber nicht gesetzt

		auto tmp = regman.alloc_temp32();
		c.MOVI2R(tmp, bb_adr);
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
		c.MOVI2R(tmp, bb_r15);
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, R[15]));
		c.MOVI2R(tmp, bb_next_instruction);
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction));
		regman.free_temp32(tmp);

		/*if (instr_uses_r15(opcode))
		{
			regman.load_reg(15);
			c.MOVI2R(regman.map_reg32(15), bb_r15);
		}*/

		if (instr_is_conditional(opcode))
		{
			regman.next_instruction(false);
			// push code into branched_code_buf
			auto current_code_ptr = c.GetWritableCodePtr();
			c.SetCodePtr(branched_code_buf);

			regman.load_cpsr();
			c._MSR(FIELD_NZCV, EncodeRegTo64(RCPSR));
			auto skip_matches = c.B((CCFlags)(CONDITION(opcode)^1));

			emit_armop_call(opcode);

			if (cycles == 0)
			{
				c.SUB(Rcycles, Rcycles, 1);
				c.ADD(Rtotal_cycles, Rtotal_cycles, Rcycles);
			}
			else if (cycles > 0)
				c.SUB(Rtotal_cycles, Rtotal_cycles, 1);

			c.SetJumpTarget(skip_matches);

			auto branched_code_size = c.GetCodePtr() - branched_code_buf;
			c.SetCodePtr(current_code_ptr);
			// register loads
			regman.load_dependencies();

			// emit branched_code_buf 
			memcpy(c.GetWritableCodePtr(), branched_code_buf, branched_code_size);
			c.SetCodePtr(c.GetWritableCodePtr() + branched_code_size);
		}
		else
		{
			regman.next_instruction(true);

			emit_armop_call(opcode);

			if (cycles == 0)
				c.ADD(Rtotal_cycles, Rtotal_cycles, Rcycles);
		}
		regman.flush_regs();
		regman.reset();

		interpreted_cycles += op_decode[PROCNUM][bb_thumb]();
	}

	auto tmp = regman.alloc_temp32();
	if (!instr_is_branch(opcode))
	{
		bb_adr += bb_opcodesize;
		c.MOVI2R(tmp, bb_adr);
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
		c.MOVI2R(tmp, bb_r15);
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, R[15]));
		c.MOVI2R(tmp, bb_next_instruction);
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction));
	}
	else
	{
		c.LDR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction));
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
	}
	regman.free_temp32(tmp);

	regman.flush_regs();
	regman.save_cpsr();

	/*if(!instr_does_prefetch(opcode))
	{
		JIT_COMMENT("!instr_does_prefetch: copy next_instruction (%08X) to instruct_adr (%08X)", cpu->next_instruction, cpu->instruct_adr);
		auto tmp = regman.alloc_temp32();
		auto rcpu = regman.alloc_temp64();
		c.MOVP2R(rcpu, &ARMPROC);
		c.LDR(INDEX_UNSIGNED, tmp, rcpu, offsetof(armcpu_t, next_instruction));
		c.STR(INDEX_UNSIGNED, tmp, rcpu, offsetof(armcpu_t, instruct_adr));
		regman.free_temp32(tmp);
		regman.free_temp64(rcpu);
		//c.mov(cpu_ptr(instruct_adr), bb_adr);
		//c.mov(cpu_ptr(instruct_adr), bb_next_instruction);
	}*/

	JIT_COMMENT("total cycles (block)");

	/*if (bb_constant_cycles > 0)
		c.ADD(Rtotal_cycles, Rtotal_cycles, bb_constant_cycles);*/

#if (PROFILER_JIT_LEVEL > 1)
	JIT_COMMENT("*** profiler - cycles");
	u32 padr = ((start_adr & 0x07FFFFFE) >> 1);
	bb_profiler_entry = c.newGpVar(kX86VarTypeGpz);
	c.mov(bb_profiler_entry, (uintptr_t)&profiler_entry[PROCNUM][padr]);
	c.add(dword_ptr(bb_profiler_entry, offsetof(PROFILER_ENTRY, cycles)), bb_total_cycles);
	profiler_entry[PROCNUM][padr].addr = start_adr;
#endif

	//c.ADDI2R(W0, Rtotal_cycles, bb_constant_cycles, W1);
	//regman.call(X2, print_out, false);
	//c.QuickCallFunction(X2, print_out);
	c.ADD(W0, Rtotal_cycles, bb_constant_cycles);
	c.ABI_PopRegisters(stashed_regs);
	c.RET();
	
#if LOG_JIT
	fprintf(stderr, "cycles %d%s\n", bb_constant_cycles, has_variable_cycles ? " + variable" : "");
#endif
	/*c.endFunc();

	ArmOpCompiled f = (ArmOpCompiled)c.make();
	if(c.getError())
	{
		fprintf(stderr, "JIT error at %s%c-%08X: %s\n", bb_thumb?"THUMB":"ARM", PROCNUM?'7':'9', start_adr, getErrorString(c.getError()));
		f = op_decode[PROCNUM][bb_thumb];
	}*/
#if LOG_JIT
	uintptr_t baddr = (uintptr_t)f;
	fprintf(stderr, "Block address %08lX\n\n", baddr);
	fflush(stderr);
#endif

	//printf("finished %d interpreted cycles\n", interpreted_cycles);
	
	libnx::jitTransitionToExecutable(&jit_page);

	JIT_COMPILED_FUNC(start_adr, PROCNUM) = (uintptr_t)f;
	return interpreted_cycles;
}

template<int PROCNUM> u32 arm_jit_compile()
{
	*PROCNUM_ptr = PROCNUM;

	// prevent endless recompilation of self-modifying code, which would be a memleak since we only free code all at once.
	// also allows us to clear compiled_funcs[] while leaving it sparsely allocated, if the OS does memory overcommit.
	u32 adr = cpu->instruct_adr;
	u32 mask_adr = (adr & 0x07FFFFFE) >> 4;
	if(((recompile_counts[mask_adr >> 1] >> 4*(mask_adr & 1)) & 0xF) > 8)
	{
		ArmOpCompiled f = op_decode[PROCNUM][cpu->CPSR.bits.T];
		JIT_COMPILED_FUNC(adr, PROCNUM) = (uintptr_t)f;
		return f();
	}
	recompile_counts[mask_adr >> 1] += 1 << 4*(mask_adr & 1);
	
	//printf("miau %d\n", (Jit_Size / 4) - (((u32*)c.GetCodePtr()) - jit_rw_addr));
	if((Jit_Size / 4) - (((u32*)c.GetCodePtr()) - jit_rw_addr) < 1000)
	{
		arm_jit_reset(true);
	}

	return compile_basicblock<PROCNUM>();
}

template u32 arm_jit_compile<0>();
template u32 arm_jit_compile<1>();

void arm_jit_reset(bool enable, bool suppress_msg)
{
#if LOG_JIT
	c.setLogger(&logger);
	freopen("desmume_jit.log", "w", stderr);
#endif
#ifdef HAVE_STATIC_CODE_BUFFER
	scratchptr = scratchpad;
#endif
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

		if (jit_rw_addr == nullptr)
		{
			libnx::jitCreate(&jit_page, Jit_Size);
			jit_rw_addr = (u32*)libnx::jitGetRwAddr(&jit_page);
			jit_rx_addr = (u32*)libnx::jitGetRxAddr(&jit_page);

			regman.c = &c;
		}
		c.SetCodePtr((u8*)jit_rw_addr);

		if (branched_code_buf == nullptr)
		{
			branched_code_buf = (u8*)malloc(64 * 4); // 64 instructions
		}
	}

#if (PROFILER_JIT_LEVEL > 0)
	reconstruct(&profiler_counter[0]);
	reconstruct(&profiler_counter[1]);
#if (PROFILER_JIT_LEVEL > 1)
	for (u8 t = 0; t < 2; t++)
	{
		for (u32 i = 0; i < (1<<26); i++)
			memset(&profiler_entry[t][i], 0, sizeof(PROFILER_ENTRY));
	}
#endif
#endif
}

#if (PROFILER_JIT_LEVEL > 0)
static int pcmp(PROFILER_COUNTER_INFO *info1, PROFILER_COUNTER_INFO *info2)
{
	return (int)(info2->count - info1->count);
}

#if (PROFILER_JIT_LEVEL > 1)
static int pcmp_entry(PROFILER_ENTRY *info1, PROFILER_ENTRY *info2)
{
	return (int)(info1->cycles - info2->cycles);
}
#endif
#endif

void arm_jit_close()
{
	printf("jit close\n");
	if (jit_rw_addr != nullptr) libnx::jitClose(&jit_page);
	if (branched_code_buf != nullptr) free(branched_code_buf);
#if (PROFILER_JIT_LEVEL > 0)
	printf("Generating profile report...");

	for (u8 proc = 0; proc < 2; proc++)
	{
		extern GameInfo gameInfo;
		u16 last[2] = {0};
		PROFILER_COUNTER_INFO *arm_info = NULL;
		PROFILER_COUNTER_INFO *thumb_info = NULL;
		
		arm_info = new PROFILER_COUNTER_INFO[4096];
		thumb_info = new PROFILER_COUNTER_INFO[1024];
		memset(arm_info, 0, sizeof(PROFILER_COUNTER_INFO) * 4096);
		memset(thumb_info, 0, sizeof(PROFILER_COUNTER_INFO) * 1024);

		// ARM
		last[0] = 0;
		for (u16 i=0; i < 4096; i++)
		{
			u16 t = 0;
			if (profiler_counter[proc].arm_count[i] == 0) continue;

			for (t = 0; t < last[0]; t++)
			{
				if (strcmp(arm_instruction_names[i], arm_info[t].name) == 0)
				{
					arm_info[t].count += profiler_counter[proc].arm_count[i];
					break;
				}
			}
			if (t == last[0])
			{
				strcpy(arm_info[last[0]++].name, arm_instruction_names[i]);
				arm_info[t].count = profiler_counter[proc].arm_count[i];
			}
		}

		// THUMB
		last[1] = 0;
		for (u16 i=0; i < 1024; i++)
		{
			u16 t = 0;
			if (profiler_counter[proc].thumb_count[i] == 0) continue;

			for (t = 0; t < last[1]; t++)
			{
				if (strcmp(thumb_instruction_names[i], thumb_info[t].name) == 0)
				{
					thumb_info[t].count += profiler_counter[proc].thumb_count[i];
					break;
				}
			}
			if (t == last[1])
			{
				strcpy(thumb_info[last[1]++].name, thumb_instruction_names[i]);
				thumb_info[t].count = profiler_counter[proc].thumb_count[i];
			}
		}

		std::qsort(arm_info, last[0], sizeof(PROFILER_COUNTER_INFO), (int (*)(const void *, const void *))pcmp);
		std::qsort(thumb_info, last[1], sizeof(PROFILER_COUNTER_INFO), (int (*)(const void *, const void *))pcmp);

		char buf[MAX_PATH] = {0};
		sprintf(buf, "desmume_jit%c_counter.profiler", proc==0?'9':'7');
		FILE *fp = fopen(buf, "w");
		if (fp)
		{
			if (!gameInfo.isHomebrew())
			{
				fprintf(fp, "Name:   %s\n", gameInfo.ROMname);
				fprintf(fp, "Serial: %s\n", gameInfo.ROMserial);
			}
			else
				fprintf(fp, "Homebrew\n");
			fprintf(fp, "CPU: ARM%c\n\n", proc==0?'9':'7');

			if (last[0])
			{
				fprintf(fp, "========================================== ARM ==========================================\n");
				for (int i=0; i < last[0]; i++)
					fprintf(fp, "%30s: %20ld\n", arm_info[i].name, arm_info[i].count);
				fprintf(fp, "\n");
			}
			
			if (last[1])
			{
				fprintf(fp, "========================================== THUMB ==========================================\n");
				for (int i=0; i < last[1]; i++)
					fprintf(fp, "%30s: %20ld\n", thumb_info[i].name, thumb_info[i].count);
				fprintf(fp, "\n");
			}

			fclose(fp);
		}

		delete [] arm_info; arm_info = NULL;
		delete [] thumb_info; thumb_info = NULL;

#if (PROFILER_JIT_LEVEL > 1)
		sprintf(buf, "desmume_jit%c_entry.profiler", proc==0?'9':'7');
		fp = fopen(buf, "w");
		if (fp)
		{
			u32 count = 0;
			PROFILER_ENTRY *tmp = NULL;

			fprintf(fp, "Entrypoints (cycles):\n");
			tmp = new PROFILER_ENTRY[1<<26];
			memset(tmp, 0, sizeof(PROFILER_ENTRY) * (1<<26));
			for (u32 i = 0; i < (1<<26); i++)
			{
				if (profiler_entry[proc][i].cycles == 0) continue;
				memcpy(&tmp[count++], &profiler_entry[proc][i], sizeof(PROFILER_ENTRY));
			}
			std::qsort(tmp, count, sizeof(PROFILER_ENTRY), (int (*)(const void *, const void *))pcmp_entry);
			if (!gameInfo.isHomebrew())
			{
				fprintf(fp, "Name:   %s\n", gameInfo.ROMname);
				fprintf(fp, "Serial: %s\n", gameInfo.ROMserial);
			}
			else
				fprintf(fp, "Homebrew\n");
			fprintf(fp, "CPU: ARM%c\n\n", proc==0?'9':'7');

			while ((count--) > 0)
				fprintf(fp, "%08X: %20ld\n", tmp[count].addr, tmp[count].cycles);

			delete [] tmp; tmp = NULL;

			fclose(fp);
		}
#endif
	}
	printf(" done.\n");
#endif
}
