#include "types.h"

#include <array>
#include <map>
#include <vector>

#include "instructions.h"
#include "instruction_attributes.h"
#include "MMU.h"
#include "MMU_timing.h"
#include "arm_jit.h"
#include "bios.h"
#include "armcpu.h"
#include "utils/bits.h"

#include "emitter/Arm64Emitter.h"
using namespace Arm64Gen;

namespace libnx {
    #include <switch.h>
}

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

#define JIT_BLOCK_SIZE (0x100000 * 8) // 8 MB

static u8 recompile_counts[(1<<26)/16];

static ARM64XEmitter c;
static libnx::Jit jit_block;
static u8* jit_rw_addr = nullptr, *jit_rx_addr = nullptr;

static const ARM64Reg RCPU = X28;
static const ARM64Reg Rtotal_cycles = W27;
static const ARM64Reg RCPSR = W26;

static u32 constant_branch = 0;

//#define JIT_DEBUG
#ifdef JIT_DEBUG
struct reg_set
{
	u32 instr;
	u32 hash;
	u32 R[15] = {0};
	u32 cpsr = 0;
	reg_set* next = nullptr;

	u32 gen_hash()
	{
		hash = 17;
		hash += hash ^ cpsr;
		for (int i = 0; i < 15; i++)
			hash += hash ^ R[15];
	}
};
static std::map<u32, reg_set*> expected_results;
static reg_set* results = nullptr;
static int results_count = 0;

static reg_set derive_state(armcpu_t* cpu, u32 instr)
{
	reg_set res;
	for (int i = 0; i < 15; i++)
		res.R[i] = cpu->R[i];
	res.cpsr = cpu->CPSR.val;
	res.instr = instr;
	return res;
}

static void record_result(armcpu_t* cpu, u32 instr, u32 pc)
{
	auto state = derive_state(cpu, instr);
	state.gen_hash();
	
	auto* last_state = expected_results[pc];
	while (last_state != nullptr)
	{
		if (last_state->hash == state.hash)
			return;
		last_state = last_state->next;
	}

	last_state->next = results + results_count;
	*last_state->next = state;
}

static void check_result(armcpu_t* cpu, u32 instr, u32 pc)
{
	auto current_state = derive_state(cpu, instr);
	current_state.gen_hash();

	auto* state = expected_results[pc];
	bool matched = false;
	while(state != nullptr) {
		matched |= state->hash != current_state.hash;
		state = state->next;
	}

	if (!matched)
		printf("unexpected result for instr %x at pc %x\n", instr, pc);
}
#endif

#define INSTR_CTX(const_cycles, proc) ((const_cycles) | (proc) << 3)
#define INSTR_CTX_CYCLES(info) ((info) & 0x7)
#define INSTR_CTX_PROC(info) (((info) >> 3) & 1)

//#define JIT_IMM_PRINT(msg, ...) printf(msg, ##__VA_ARGS__);
#define JIT_IMM_PRINT(msg, ...)

static class {
    ARM64Reg mapping[16];
    int reg_usage[16];
    u32 next_unassigned = 0;
    u32 regs_dirty = 0;
	u32 temps = 0;

    void inc_usage()
    {
        for (int i = 0; i < 16; i++)
            reg_usage[i]++;
    }
    int reg_least_used()
    {
        int least_used_reg = -1;
        int least_used_usage = -1;
        for (int i = 0; i < 16; i++)
        {
            if (mapping[i] != INVALID_REG && reg_usage[i] > least_used_usage) {
                least_used_usage = reg_usage[i];
                least_used_reg = i;
            }
        }
        return least_used_reg;
    }

public:
    void reset()
    {
		next_unassigned = 0;
		regs_dirty = 0;
        for (int i = 0; i < 16; i++)
		{
            mapping[i] = INVALID_REG;
			reg_usage[i] = 0;
		}
    }

	bool is_loaded(int reg)
	{
		return mapping[reg] != INVALID_REG;
	}

	ARM64Reg new_temp()
	{
		const std::array<ARM64Reg, 7> allocation_order{ {
                W9,
                W10,
                W11,
                W12,
                W13,
                W14,
                W15
		} };
		for (u32 i = 0; i < allocation_order.size(); i++)
		{
			auto reg = allocation_order[i];
			if (!(temps & BIT(reg)))
			{
				temps |= BIT(reg);
				return reg;
			}
		}

		printf("All temp registers used\n");
		abort();
	}

	void free_temp(ARM64Reg reg)
	{
		temps &= ~BIT(reg);
	}

    ARM64Reg get(u32 reg, bool write = false)
    {
        inc_usage();
        reg_usage[reg] = 0;
        
		bool load = false;
        if (mapping[reg] == INVALID_REG)
        {
			load = true;
            const std::array<ARM64Reg, 7> allocation_order{ {
                W19,
                W20,
                W21,
                W22,
                W23,
                W24,
                W25
            } };
            
            if (next_unassigned < allocation_order.size()) // we have free regs
                mapping[reg] = allocation_order[next_unassigned++];
            else // repurpose an already assigned
            {
                auto least_used = reg_least_used();
                mapping[reg] = mapping[least_used];
				if (regs_dirty & BIT(least_used))
				{
                	c.STR(INDEX_UNSIGNED, mapping[least_used], RCPU, offsetof(armcpu_t, R) + least_used * 4);
					regs_dirty &= ~BIT(least_used);
				}
                mapping[least_used] = INVALID_REG;
				reg_usage[least_used] = 0;
            }

			c.LDR(INDEX_UNSIGNED, mapping[reg], RCPU, offsetof(armcpu_t, R) + reg * 4);
        }
		JIT_IMM_PRINT("%s reg %d into %d(%d)\n", load ? "loading" : "get", reg, mapping[reg], write);

        regs_dirty |= write << reg;

        return mapping[reg];
    }

	void flush_regs()
	{
		for (int i = 0; i < 16; i++)
		{
			if (regs_dirty & BIT(i)) {
				c.STR(INDEX_UNSIGNED, mapping[i], RCPU, offsetof(armcpu_t, R) + i * 4);
				JIT_IMM_PRINT("flush back %d into %d\n", mapping[i], i);
			}
		}
		regs_dirty = 0;
	}

	template <typename T>
	void call(T f)
	{
		c.ABI_PushRegisters(BitSet32(temps));
		c.QuickCallFunction(X7, f);
		c.ABI_PopRegisters(BitSet32(temps));
	}
} regman;

static bool cpsr_dirty = false;
static u32 nzcv_location = 0; // set bit = stored in native register

#define NZCV_KEPT 1
#define NZCV_TRASH_ALL (0xfu<<28)
#define NZCV_NZCV_USEFUL 0
#define NZCV_NZ_USEFUL (3u<<28)

// set bit = won't contain something useful
static void update_nzcv(u32 trashed)
{
	auto tmp = regman.new_temp();
	for (u32 i = 0; i < 4; i++)
	{
		u32 bit = trashed & (1u << (28 + i));
		if (nzcv_location & bit)
		{
			// we need to retrive it
			const CCFlags flags[] = {CC_VS, CC_CS, CC_EQ, CC_MI};
			c.CSET(tmp, flags[i]);
			c.BFI(RCPSR, tmp, 28 + i, 1);
		}
	}
	regman.free_temp(tmp);
	nzcv_location = ~trashed;
}

static FixupBranch branch_on_nzcv(u32 cond, u32 nzcv)
{
	const u32 required_bits[] = {BIT(30), BIT(29),
		(u32)BIT(31), BIT(28), BIT(29)|BIT(30), 
		(u32)BIT(31)|BIT(28), 0xfu<<28};

	u32 required = required_bits[cond>>1];
	if ((nzcv & required) == required)
		return c.B((CCFlags)(cond^1));
	/* somehow doesn't work :(
	if (cond < 8) // < 8 means only a single bit
	{
		JIT_IMM_PRINT("branched from RCPSR %d %d\n", cond, __builtin_ffs(required));
		auto tmp = regman.new_temp();
		c.UBFX(tmp, RCPSR, __builtin_ffs(required) - 1, 1);
		regman.free_temp(tmp); // the register is still safe until an alloc happens
		if (cond & 1)
			return c.CBZ(tmp);
		else
			return c.CBNZ(tmp);
	}*/
	update_nzcv(NZCV_TRASH_ALL);
	c._MSR(FIELD_NZCV, EncodeRegTo64(RCPSR));
	return c.B((CCFlags)(cond^1));
}

static void load_cpsr()
{
	nzcv_location = 0;
	c.LDR(INDEX_UNSIGNED, RCPSR, RCPU, offsetof(armcpu_t, CPSR.val)); 
}
static void save_cpsr()
{
	if (cpsr_dirty)
	{
		update_nzcv(NZCV_TRASH_ALL);
		c.STR(INDEX_UNSIGNED, RCPSR, RCPU, offsetof(armcpu_t, CPSR.val));
		cpsr_dirty = false;
	}
}

template<int PROCNUM, int thumb>
static u32 FASTCALL OP_DECODE(u32 cycles, u32 instrs_num)
{
    auto cpu = &ARMPROC;
	u32 adr = cpu->instruct_adr;
	if(thumb)
	{
		cpu->next_instruction = adr + 2;
		cpu->R[15] = adr + 4;
		u32 opcode = _MMU_read16<PROCNUM, MMU_AT_CODE>(adr);
		cycles += thumb_instructions_set[PROCNUM][opcode>>6](opcode);
	}
	else
	{
		cpu->next_instruction = adr + 4;
		cpu->R[15] = adr + 8;
		u32 opcode = _MMU_read32<PROCNUM, MMU_AT_CODE>(adr);
		if(CONDITION(opcode) == 0xE || TEST_COND(CONDITION(opcode), CODE(opcode), cpu->CPSR))
			cycles += arm_instructions_set[PROCNUM][INSTRUCTION_INDEX(opcode)](opcode);
		else
			cycles += 1;
	}
	cpu->instruct_adr = cpu->next_instruction;
	return cycles;
}

static const ArmOpCompiled op_decode[2][2] = { OP_DECODE<0,0>, OP_DECODE<0,1>, OP_DECODE<1,0>, OP_DECODE<1,1> };

static bool instr_is_conditional(bool thumb, u32 opcode)
{
	if(thumb) return false;
	
	return !(CONDITION(opcode) == 0xE
	         || (CONDITION(opcode) == 0xF && CODE(opcode) == 5));
}

static void prefetch_addrs(u32 addr, u32 opcode_size)
{
	auto tmp = regman.new_temp();
	c.MOVI2R(tmp, addr);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
	c.ADD(tmp, tmp, opcode_size);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction));
	c.ADD(tmp, tmp, opcode_size);
	c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, R[15]));
	regman.free_temp(tmp);
}

typedef u32 (*ArmOpCompiler)(u32 pc, u32 instr, u32 ctx);

/*
	Cycle counting:
		The cycles are counted with the Rtotal_cycles register as well 
		as the constant_cycles variable. If the instruction cycles are fully dynamic
		or the instruction is conditional one is added to constant_cycles and
		the instruction is responsible for adding it's cycles minus one
		to Rtotal_cycles.

*/

/*
	hello preprocessor my old friend,
	I've come to try to write a JIT again,
	because I really want to play Pokemon,
	left it's seed while I was younger,

	sometimes jokes in comments become worse everytime you open the source file...
*/


#define ARM_IMPL_BRANCH(branched_code, nzcv_override) \
	auto branches = instr_is_conditional(false, instr); \
	FixupBranch branch; \
	bool force_retrive_nzcv; \
	if (branches) \
	{ \
		u32 prev_nzcv_location = nzcv_location; \
		force_retrive_nzcv = nzcv_override != 1/* || ~((u32)nzcv_override) != nzcv_location*/; \
		if (force_retrive_nzcv) \
			update_nzcv(NZCV_TRASH_ALL); \
		branch = branch_on_nzcv(CONDITION(instr), prev_nzcv_location); \
	} \
	else if (nzcv_override != 1) \
		update_nzcv(nzcv_override); \
	branched_code \
	if (branches) \
	{ \
		if (force_retrive_nzcv) \
		{ \
			nzcv_location = ~nzcv_override; \
			update_nzcv(NZCV_TRASH_ALL); \
		} \
		if (INSTR_CTX_CYCLES(ctx) > 1) \
			c.ADD(Rtotal_cycles, Rtotal_cycles, INSTR_CTX_CYCLES(ctx) - 1); \
		c.SetJumpTarget(branch); \
	}

#define ARM_SHIFT_IMM \
	auto rm = regman.get(REG_POS(instr, 0)); \
	auto shift = (instr>>7)&0x1f;

#define ARM_IMM_VAL \
	auto shift = (ARM64Reg)((instr>>7)&0x1e); \
	auto imm = ROR((instr&0xFF), shift);

#define ARM_SHIFT_REG \
	auto rm = regman.get(REG_POS(instr, 0)); \
	auto rs = regman.get(REG_POS(instr, 8));

#define ARM_SETUP_C_NOP(skind)

#define ARM_SETUP_C_SHIFT_IMM(skind) \
	auto rcf = WZR; \
	auto copy_c = false; \
	if (!(skind == ST_LSL && shift == 0)) \
	{ \
		rcf = regman.new_temp(); \
		copy_c = true; \
		if (skind == ST_LSL) \
			c.UBFX(rcf, rm, 32 - shift, 1); \
		else \
			c.UBFX(rcf, rm, shift?shift-1:(skind==ST_ROR?0:31), 1); \
	}
#define ARM_SETUP_C_IMM_VAL(discard) \
	auto rcf = WZR; \
	auto copy_c = false; \
	if (shift) \
	{ \
		rcf = regman.new_temp(); \
		copy_c = true; \
		c.MOVZ(rcf, BIT31(imm)); \
	}

#define ARM_SETUP_SHIFT_REG(skind) \
	auto op2 = regman.new_temp(); \
	c.UBFX(op2, rs, 0, 8); \
	if (skind != ST_ROR) \
	{ \
		c.CMP(op2, 32); \
		if(skind == ST_LSL)  \
			c.LSLV(op2, rm, op2); \
		else if (skind == ST_LSR) \
			c.LSRV(op2, rm, op2); \
		else /* ST_ASR */ \
			c.ASRV(op2, rm, op2); \
		if (skind == ST_ASR) \
		{ \
			auto ge_than_type = regman.new_temp(); \
			c.MOV(ge_than_type, rm, ArithOption(ge_than_type, ST_ASR, 31)); \
			c.CSEL(op2, ge_than_type, op2, CC_GE); \
			regman.free_temp(ge_than_type); \
		} \
		else \
			c.CSEL(op2, WZR, op2, CC_GE); \
	} \
	else \
		c.RORV(op2, rm, op2);

#define ARM_SETUP_SHIFT_REG_S(skind) \
	auto rcf = regman.new_temp(); \
	auto copy_c = true; \
	c.UBFX(rcf, RCPSR, 29, 1); \
	auto op2 = regman.new_temp(); \
	c.MOV(op2, rm); \
	auto shift = regman.new_temp(); \
	c.UBFX(shift, rs, 0, 8); \
	auto __zero = c.CBZ(shift); \
	if (skind != ST_ROR) \
	{ \
		c.CMP(shift, 32); \
		c.SUB(shift, shift, 1); \
		if (skind == ST_LSL)  \
			c.LSLV(op2, rm, shift); \
		else if (skind == ST_LSR) \
			c.LSRV(op2, rm, shift); \
		else /* ST_ASR */ \
			c.ASRV(op2, rm, shift); \
		auto ge_than_type = regman.new_temp(); \
		c.MOV(ge_than_type, skind==ST_ASR?rm:WZR, ArithOption(ge_than_type, ST_ASR, 31)); \
		c.UBFX(rcf, op2, skind==ST_LSL?31:0, 1); \
		c.CSEL(rcf, ge_than_type, rcf, CC_GT); \
		c.MOV(op2, op2, ArithOption(op2, skind, 1)); \
		c.CSEL(op2, ge_than_type, op2, CC_GE); \
		regman.free_temp(ge_than_type); \
	} \
	else \
	{ \
		c.SUB(shift, shift, 1); \
		c.RORV(op2, rm, shift); \
		c.UBFX(rcf, op2, 0, 1); \
		c.ROR(op2, op2, 1); \
	} \
	regman.free_temp(shift); \
	c.SetJumpTarget(__zero);

#define ARM_ALU_SHIFT_IMM_REV(a64inst, move) \
	auto shifted = regman.new_temp(); \
	move \
	c.a64inst(rd, shifted, rn); \
	regman.free_temp(shifted);

#define ARM_ALU_LSL_REV(a64inst) ARM_ALU_SHIFT_IMM_REV(a64inst, c.MOV(shifted, rm, ArithOption(shifted, ST_LSL, shift));)
#define ARM_ALU_LSR_REV(a64inst) ARM_ALU_SHIFT_IMM_REV(a64inst, c.MOV(shifted, shift?rm:WZR, ArithOption(rd, ST_LSR, shift));)
#define ARM_ALU_ASR_REV(a64inst) ARM_ALU_SHIFT_IMM_REV(a64inst, c.MOV(shifted, rm, ArithOption(rd, ST_ASR, shift?shift:31));)

#define ARM_ALU_LSL_SIMPLE(a64inst) c.a64inst(rd, rn, rm, ArithOption(rd, ST_LSL, shift));
#define ARM_ALU_LSR_SIMPLE(a64inst) c.a64inst(rd, rn, shift?rm:WZR, ArithOption(rd, ST_LSR, shift));
#define ARM_ALU_ASR_SIMPLE(a64inst) c.a64inst(rd, rn, rm, ArithOption(rd, ST_ASR, shift?shift:31));

#define ARM_ALU_ROR_SIMPLE(a64inst) \
	if (shift) \
		c.a64inst(rd, rn, rm, ArithOption(rd, ST_ROR, shift)); \
	else \
	{ \
		auto op2 = regman.new_temp(); \
		c.MOV(op2, rm); \
		c.BFM(op2, RCPSR, 29, 1); \
		c.a64inst(rd, rn, op2, ArithOption(rd, ST_ROR, 1)); \
		regman.free_temp(op2); \
	}

#define ARM_ALU_ROR_MANUALLY(a64inst) \
	auto op2 = regman.new_temp(); \
	if (!shift) \
	{ \
		c.MOV(op2, rm); \
		c.BFM(op2, RCPSR, 29, 1); \
		c.ROR(op2, op2, 1); \
	} \
	else \
		c.ROR(op2, rm, shift); \
	c.a64inst(rd, rn, op2); \
	regman.free_temp(op2);

#define ARM_ALU_ROR_REV(a64inst) \
	auto op2 = regman.new_temp(); \
	if (!shift) \
	{ \
		c.MOV(op2, rm); \
		c.BFM(op2, RCPSR, 29, 1); \
		c.ROR(op2, op2, 1); \
	} \
	else \
		c.ROR(op2, rm, shift); \
	c.a64inst(rd, op2, rn); \
	regman.free_temp(op2);

#define ARM_ALU_IMM_VAL_SIMPLE(a64inst) \
	auto tmp = regman.new_temp(); \
	c.a64inst##I2R(rd, rn, imm, tmp); \
	regman.free_temp(tmp);

#define ARM_ALU_IMM_VAL_MANUALLY(a64inst) \
	auto op2 = regman.new_temp(); \
	c.MOVI2R(op2, imm); \
	c.a64inst(rd, rn, op2); \
	regman.free_temp(op2);

#define ARM_ALU_IMM_VAL_REV(a64inst) \
	auto op2 = regman.new_temp(); \
	c.MOVI2R(op2, imm); \
	c.a64inst(rd, op2, rn); \
	regman.free_temp(op2);

#define ARM_ALU_SHIFT_REG_SIMPLE(a64inst) \
	c.a64inst(rd, rn, op2);
#define ARM_ALU_SHIFT_REG_REV(a64inst) \
	c.a64inst(rd, op2, rn);

#define ARM_ALU_COPY_NZCV \
	/*auto nzcv = regman.new_temp(); \
	c.MRS(EncodeRegTo64(nzcv), FIELD_NZCV); \
	c.UBFX(nzcv, nzcv, 28, 4); \
	c.BFI(RCPSR, nzcv, 28, 4); \
	regman.free_temp(nzcv);*/ \
	cpsr_dirty = true;

#define ARM_ALU_TST_RD \
	c.TST(rd, rd);

#define ARM_ALU_COPY_NZC \
	/*auto nzcv = regman.new_temp(); \
	c.MRS(EncodeRegTo64(nzcv), FIELD_NZCV); \
	c.UBFX(nzcv, nzcv, 30, 2); \
	c.BFI(RCPSR, nzcv, 30, 2); \
	regman.free_temp(nzcv);*/ \
	if (copy_c) \
	{ \
		c.BFI(RCPSR, rcf, 29, 1); \
		regman.free_temp(rcf); \
	} \
	cpsr_dirty = true;

#define ARM_ARITHMETIC_OP(name, preamble, body, nzcv_override) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto rd = regman.get(REG_POS(instr, 12), true); \
		auto rn = regman.get(REG_POS(instr, 16)); \
		preamble \
		if (REG_POS(instr, 12) == 15) return 0; \
		ARM_IMPL_BRANCH(body, nzcv_override) \
		return 1; \
	}
#define ARM_MOV_OP(name, preamble, body, nzcv_override) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto rd = regman.get(REG_POS(instr, 12), true); \
		auto rn = WZR; \
		preamble \
		if (REG_POS(instr, 12) == 15) return 0; \
		ARM_IMPL_BRANCH(body, nzcv_override) \
		return 1; \
	}
#define ARM_CMP_OP(name, preamble, body, nzcv_override) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto rd = regman.new_temp(); \
		auto rn = regman.get(REG_POS(instr, 16)); \
		preamble \
		ARM_IMPL_BRANCH(body, nzcv_override) \
		regman.free_temp(rd); \
		return 1; \
	}

#define ARM_ALU_IMPL_SIMPLE(fdef, a64inst, name, lsl, lsr, asr, ror, imm_val, shift_reg) \
	ARM_##fdef##_OP(name##_LSL_IMM, ARM_SHIFT_IMM, ARM_ALU_##lsl(a64inst), NZCV_KEPT) \
	ARM_##fdef##_OP(name##_LSR_IMM, ARM_SHIFT_IMM, ARM_ALU_##lsr(a64inst), NZCV_KEPT) \
	ARM_##fdef##_OP(name##_ASR_IMM, ARM_SHIFT_IMM, ARM_ALU_##asr(a64inst), NZCV_KEPT) \
	ARM_##fdef##_OP(name##_ROR_IMM, ARM_SHIFT_IMM, ARM_ALU_##ror(a64inst), NZCV_KEPT) \
	ARM_##fdef##_OP(name##_IMM_VAL, ARM_IMM_VAL, ARM_ALU_##imm_val(a64inst), NZCV_KEPT) \
	ARM_##fdef##_OP(name##_LSL_REG, ARM_SHIFT_REG, ARM_SETUP_SHIFT_REG(ST_LSL) ARM_ALU_##shift_reg(a64inst) regman.free_temp(op2);, NZCV_TRASH_ALL) \
	ARM_##fdef##_OP(name##_LSR_REG, ARM_SHIFT_REG, ARM_SETUP_SHIFT_REG(ST_LSR) ARM_ALU_##shift_reg(a64inst) regman.free_temp(op2);, NZCV_TRASH_ALL) \
	ARM_##fdef##_OP(name##_ASR_REG, ARM_SHIFT_REG, ARM_SETUP_SHIFT_REG(ST_ASR) ARM_ALU_##shift_reg(a64inst) regman.free_temp(op2);, NZCV_TRASH_ALL) \
	ARM_##fdef##_OP(name##_ROR_REG, ARM_SHIFT_REG, ARM_SETUP_SHIFT_REG(ST_ROR) ARM_ALU_##shift_reg(a64inst) regman.free_temp(op2);, NZCV_TRASH_ALL)

#define ARM_ALU_IMPL_SIMPLE_S(fdef, a64inst, name, lsl, lsr, asr, ror, imm_val, shift_reg, prepare_flags, prepare_flags_imm, prepare_flags_sreg, copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_LSL_IMM, ARM_SHIFT_IMM, \
		prepare_flags(ST_LSL) \
		ARM_ALU_##lsl(a64inst) \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_LSR_IMM, ARM_SHIFT_IMM, \
		prepare_flags(ST_LSR) \
		ARM_ALU_##lsr(a64inst) \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_ASR_IMM, ARM_SHIFT_IMM, \
		prepare_flags(ST_ASR) \
		ARM_ALU_##asr(a64inst) \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_ROR_IMM, ARM_SHIFT_IMM, \
		prepare_flags(ST_ROR) \
		ARM_ALU_##ror(a64inst) \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_IMM_VAL, ARM_IMM_VAL, \
		prepare_flags_imm(0) \
		ARM_ALU_##imm_val(a64inst) \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_LSL_REG, ARM_SHIFT_REG, \
		prepare_flags_sreg(ST_LSL) \
		ARM_ALU_##shift_reg(a64inst) \
		regman.free_temp(op2); \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_LSR_REG, ARM_SHIFT_REG, \
		prepare_flags_sreg(ST_LSR) \
		ARM_ALU_##shift_reg(a64inst) \
		regman.free_temp(op2); \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_ASR_REG, ARM_SHIFT_REG, \
		prepare_flags_sreg(ST_ASR) \
		ARM_ALU_##shift_reg(a64inst) \
		regman.free_temp(op2); \
		copy_flags, nzcv_override) \
	ARM_##fdef##_OP(name##_ROR_REG, ARM_SHIFT_REG, \
		prepare_flags_sreg(ST_ROR) \
		ARM_ALU_##shift_reg(a64inst) \
		regman.free_temp(op2); \
		copy_flags, nzcv_override)

ARM_ALU_IMPL_SIMPLE(ARITHMETIC, AND, AND, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(ARITHMETIC, EOR, EOR, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(ARITHMETIC, ORR, ORR, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(ARITHMETIC, BIC, BIC, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_MANUALLY, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(ARITHMETIC, ADD, ADD, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_MANUALLY, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(ARITHMETIC, SUB, SUB, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_MANUALLY, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(ARITHMETIC, SUB, RSB, LSL_REV, LSR_REV, ASR_REV, ROR_REV, IMM_VAL_REV, SHIFT_REG_REV)
ARM_ALU_IMPL_SIMPLE(MOV, ORR, MOV, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE)
ARM_ALU_IMPL_SIMPLE(MOV, ORN, MVN, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_MANUALLY, SHIFT_REG_SIMPLE)

ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, ADDS, ADD_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_MANUALLY, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_NOP, ARM_SETUP_C_NOP, ARM_SETUP_SHIFT_REG, ARM_ALU_COPY_NZCV, NZCV_NZCV_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, SUBS, SUB_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_MANUALLY, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_NOP, ARM_SETUP_C_NOP, ARM_SETUP_SHIFT_REG, ARM_ALU_COPY_NZCV, NZCV_NZCV_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, SUBS, RSB_S, LSL_REV, LSR_REV, ASR_REV, ROR_REV, IMM_VAL_REV, SHIFT_REG_REV, ARM_SETUP_C_NOP, ARM_SETUP_C_NOP, ARM_SETUP_SHIFT_REG, ARM_ALU_COPY_NZCV, NZCV_NZCV_USEFUL)

ARM_ALU_IMPL_SIMPLE_S(CMP, SUBS, CMP, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_MANUALLY, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_NOP, ARM_SETUP_C_NOP, ARM_SETUP_SHIFT_REG, ARM_ALU_COPY_NZCV, NZCV_NZCV_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(CMP, ADDS, CMN, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_MANUALLY, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_NOP, ARM_SETUP_C_NOP, ARM_SETUP_SHIFT_REG, ARM_ALU_COPY_NZCV, NZCV_NZCV_USEFUL)

ARM_ALU_IMPL_SIMPLE_S(CMP, ANDS, TST, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(CMP, EOR, TEQ, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_TST_RD ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)

ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, ANDS, AND_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, EOR, EOR_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_TST_RD ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, ORR, ORR_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_TST_RD ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(ARITHMETIC, BIC, BIC_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_MANUALLY, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_TST_RD ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)

ARM_ALU_IMPL_SIMPLE_S(MOV, ORR, MOV_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_SIMPLE, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_TST_RD ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)
ARM_ALU_IMPL_SIMPLE_S(MOV, ORN, MVN_S, LSL_SIMPLE, LSR_SIMPLE, ASR_SIMPLE, ROR_SIMPLE, IMM_VAL_MANUALLY, SHIFT_REG_SIMPLE, ARM_SETUP_C_SHIFT_IMM, ARM_SETUP_C_IMM_VAL, ARM_SETUP_SHIFT_REG_S, ARM_ALU_TST_RD ARM_ALU_COPY_NZC, NZCV_NZ_USEFUL)

#define ARM_ALU_OP_DEF_ALL_NULL(T, D, N, S) \
   static const ArmOpCompiler ARM_OP_##T##_LSL_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_LSL_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_LSR_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_LSR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ASR_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ASR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ROR_IMM = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_ROR_REG = nullptr; \
   static const ArmOpCompiler ARM_OP_##T##_IMM_VAL = nullptr;
#define ARM_ALU_OP_DEF_ALL_NULLC(T, D, N, S) \
   /*static const ArmOpCompiler ARM_OP_##T##_LSL_IMM = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_LSL_REG = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_LSR_IMM = nullptr; */\
   /*static const ArmOpCompiler ARM_OP_##T##_LSR_REG = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_ASR_IMM = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_ASR_REG = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_ROR_IMM = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_ROR_REG = nullptr;*/ \
   /*static const ArmOpCompiler ARM_OP_##T##_IMM_VAL = nullptr;*/
ARM_ALU_OP_DEF_ALL_NULLC(AND  , 2, 1, false); // shift imm
ARM_ALU_OP_DEF_ALL_NULLC(AND_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(EOR  , 2, 1, false); // shift imm
ARM_ALU_OP_DEF_ALL_NULLC(EOR_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(SUB  , 2, 1, false); // shift imm + other
ARM_ALU_OP_DEF_ALL_NULLC(SUB_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(RSB  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULLC(RSB_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(ADD  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULLC(ADD_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(ADC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(ADC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(SBC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(SBC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULL(RSC  , 2, 1, false);
ARM_ALU_OP_DEF_ALL_NULL(RSC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(TST  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(TEQ  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(CMP  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(CMN  , 0, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(ORR  , 2, 1, false); // shift imm
ARM_ALU_OP_DEF_ALL_NULLC(ORR_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(MOV  , 2, 0, false);
ARM_ALU_OP_DEF_ALL_NULLC(MOV_S, 2, 0, true);
ARM_ALU_OP_DEF_ALL_NULLC(BIC  , 2, 1, false); // shift imm
ARM_ALU_OP_DEF_ALL_NULLC(BIC_S, 2, 1, true);
ARM_ALU_OP_DEF_ALL_NULLC(MVN  , 2, 0, false);
ARM_ALU_OP_DEF_ALL_NULLC(MVN_S, 2, 0, true);

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
inline u64 value_cycles(u32 value, u32 cyc){ return ((u64)value)|(((u64)(cyc))<<32); }
u64  _MMU_read08_9(u32 addr) {
	return value_cycles(_MMU_read08<0>(addr),MMU_aluMemAccessCycles<0,8,MMU_AD_READ>(3,addr)); 
}
u64  _MMU_read08_7(u32 addr) { 
	return value_cycles(_MMU_read08<1>(addr),MMU_aluMemAccessCycles<1,8,MMU_AD_READ>(3,addr)); 
}
u64 _MMU_read16_9(u32 addr) {
	return value_cycles(_MMU_read16<0>(addr&0xFFFFFFFE),MMU_aluMemAccessCycles<0,16,MMU_AD_READ>(3,addr)); 
}
u64 _MMU_read16_7(u32 addr) { 
	return value_cycles(_MMU_read16<1>(addr&0xFFFFFFFE),MMU_aluMemAccessCycles<1,16,MMU_AD_READ>(3,addr)); 
}
u64 _MMU_read32_9(u32 addr) {
	auto value = ::ROR(_MMU_read32<0>(addr&0xFFFFFFFC),8*(addr&3));
	return value_cycles(value, MMU_aluMemAccessCycles<0,32,MMU_AD_READ>(3,addr));
}
u64 _MMU_read32_7(u32 addr) {
	auto value = ::ROR(_MMU_read32<1>(addr&0xFFFFFFFC),8*(addr&3));
	return value_cycles(value,MMU_aluMemAccessCycles<1,32,MMU_AD_READ>(3,addr));
}

u32 _MMU_write08_9(u32 addr, u8  val) { 
	_MMU_write08<0>(addr, val);
	return MMU_aluMemAccessCycles<0,8,MMU_AD_WRITE>(2,addr); 
}
u32 _MMU_write08_7(u32 addr, u8  val) { 
	_MMU_write08<1>(addr, val); 
	return MMU_aluMemAccessCycles<1,8,MMU_AD_WRITE>(2,addr); 
}
u32 _MMU_write16_9(u32 addr, u16 val) { 
	_MMU_write16<0>(addr & 0xFFFFFFFE, val);
	return MMU_aluMemAccessCycles<0,16,MMU_AD_WRITE>(2,addr);
}
u32 _MMU_write16_7(u32 addr, u16 val) { 
	_MMU_write16<1>(addr & 0xFFFFFFFE, val); 
	return MMU_aluMemAccessCycles<1,16,MMU_AD_WRITE>(2,addr);
}
u32 _MMU_write32_9(u32 addr, u32 val) { 
	_MMU_write32<0>(addr & 0xFFFFFFFC, val);
	return MMU_aluMemAccessCycles<0,32,MMU_AD_WRITE>(2,addr);
}
u32 _MMU_write32_7(u32 addr, u32 val) {
	_MMU_write32<1>(addr & 0xFFFFFFFC, val);
	return MMU_aluMemAccessCycles<1,32,MMU_AD_WRITE>(2,addr);
}

static const uintptr_t mem_funcs[12] =
{
   (uintptr_t)_MMU_read08_9 , (uintptr_t)_MMU_read08_7,
   (uintptr_t)_MMU_write08_9, (uintptr_t)_MMU_write08_7,
   (uintptr_t)_MMU_read16_9,  (uintptr_t)_MMU_read16_7,
   (uintptr_t)_MMU_write16_9, (uintptr_t)_MMU_write16_7,
   (uintptr_t)_MMU_read32_9,  (uintptr_t)_MMU_read32_7,
   (uintptr_t)_MMU_write32_9, (uintptr_t)_MMU_write32_7
};

#define ARM_MEM_IMM_OFF \
	auto imm = instr & 0xfff;

#define ARM_MEM_F_IDX(write, size) \
	int size_or; \
	if (size == 1) \
		size_or = 0; \
	else if (size == 2) \
		size_or = 4; \
	else if (size == 4) \
		size_or = 8; \
	int func_idx = INSTR_CTX_PROC(ctx) | size_or | (write<<1);

#define ARM_MEM_MOVXT(dst, src, size, sign) \
	switch (size|(sign<<3)) \
	{ \
		case 1: c.UXTB(dst, src); break; \
		case 1|8: c.SXTB(dst, src); break; \
		case 2: c.UXTH(dst, src); break; \
		case 2|8: c.SXTH(dst, src); break; \
		case 4: case 4|8: c.MOV(dst, src); break; \
		default: break; \
	} \

#if 1
#define ARM_LDR_IMPL(name, preamble, calc_offset, post, writeback, size, sign) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto rd = regman.get(REG_POS(instr, 12), true); \
		auto rn = regman.get(REG_POS(instr, 16), writeback); \
		preamble \
		if (REG_POS(instr, 12) == 15 || (writeback && REG_POS(instr, 16) == 15)) return 0; \
		ARM_IMPL_BRANCH( \
			auto addr = regman.new_temp(); \
			calc_offset \
			c.MOV(W0, post ? rn : addr); \
			if (post || writeback) \
				c.MOV(rn, addr); \
			regman.free_temp(addr); \
			ARM_MEM_F_IDX(0,size) \
			regman.call(mem_funcs[func_idx]); \
			auto cycles = regman.new_temp(); \
			c.UBFX(EncodeRegTo64(cycles), X0, 32, 32); /* extract the cycles */ \
			c.ADD(Rtotal_cycles, Rtotal_cycles, cycles); \
			c.SUB(Rtotal_cycles, Rtotal_cycles, 1); \
			regman.free_temp(cycles); \
			ARM_MEM_MOVXT(rd, W0, size, sign) \
		, NZCV_TRASH_ALL) \
		return 1; \
	}
#define ARM_STR_IMPL(name, preamble, calc_offset, post, writeback, size, sign) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto rd = regman.get(REG_POS(instr, 12)); \
		auto rn = regman.get(REG_POS(instr, 16), writeback); \
		preamble \
		if (writeback && REG_POS(instr, 16) == 15) return 0; \
		ARM_IMPL_BRANCH( \
			auto addr = regman.new_temp(); \
			calc_offset \
			c.MOV(W0, post ? rn : addr); \
			if (post || writeback) \
				c.MOV(rn, addr); \
			regman.free_temp(addr); \
			ARM_MEM_MOVXT(W1, rd, size, 0) \
			ARM_MEM_F_IDX(1, size) \
			regman.call(mem_funcs[func_idx]); \
			c.ADD(Rtotal_cycles, Rtotal_cycles, W0); \
			c.SUB(Rtotal_cycles, Rtotal_cycles, 1); \
		, NZCV_TRASH_ALL) \
		return 1; \
	}
#else
#define ARM_LDR_IMPL(name, preamble, calc_offset, post, writeback, size, sign) \
	ArmOpCompiler ARM_OP_##name = nullptr;

#define ARM_STR_IMPL(name, preamble, calc_offset, post, writeback, size, sign) \
	ArmOpCompiler ARM_OP_##name = nullptr;
#endif

#define ARM_MEM_ROR \
	if (!shift) \
	{ \
		c.MOV(addr, rm); \
		c.BFM(addr, RCPSR, 29, 1); \
		c.ROR(addr, addr, 1); \
	} \
	else \
		c.ROR(addr, rm, shift); \

#define ARM_MEM_OP_SINGLE(kind, T, Q, post, writeback, size) \
	ARM_##kind##_IMPL(T##_M_LSL_##Q, ARM_SHIFT_IMM, c.SUB(addr, rn, rm, ArithOption(addr, ST_LSL, shift));, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_P_LSL_##Q, ARM_SHIFT_IMM, c.ADD(addr, rn, rm, ArithOption(addr, ST_LSL, shift));, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_M_LSR_##Q, ARM_SHIFT_IMM, c.SUB(addr, rn, shift?rm:WZR, ArithOption(addr, ST_LSR, shift));, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_P_LSR_##Q, ARM_SHIFT_IMM, c.ADD(addr, rn, shift?rm:WZR, ArithOption(addr, ST_LSR, shift));, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_M_ASR_##Q, ARM_SHIFT_IMM, c.SUB(addr, rn, rm, ArithOption(addr, ST_ASR, shift?shift:31));, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_P_ASR_##Q, ARM_SHIFT_IMM, c.ADD(addr, rn, rm, ArithOption(addr, ST_ASR, shift?shift:31));, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_M_ROR_##Q, ARM_SHIFT_IMM, ARM_MEM_ROR c.SUB(addr, rn, addr);, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_P_ROR_##Q, ARM_SHIFT_IMM, ARM_MEM_ROR c.ADD(addr, rn, addr);, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_M_##Q, ARM_MEM_IMM_OFF, c.SUB(addr, rn, imm);, post, writeback, size, 0) \
	ARM_##kind##_IMPL(T##_P_##Q, ARM_MEM_IMM_OFF, c.ADD(addr, rn, imm);, post, writeback, size, 0)

ARM_MEM_OP_SINGLE(STR, STR, IMM_OFF_PREIND, 0, 1, 4)
ARM_MEM_OP_SINGLE(STR, STR, IMM_OFF, 0, 0, 4)
ARM_MEM_OP_SINGLE(STR, STR, IMM_OFF_POSTIND, 1, 1, 4)

ARM_MEM_OP_SINGLE(LDR, LDR, IMM_OFF_PREIND, 0, 1, 4)
ARM_MEM_OP_SINGLE(LDR, LDR, IMM_OFF, 0, 0, 4)
ARM_MEM_OP_SINGLE(LDR, LDR, IMM_OFF_POSTIND, 1, 1, 4)

ARM_MEM_OP_SINGLE(STR, STRB, IMM_OFF_PREIND, 0, 1, 1)
ARM_MEM_OP_SINGLE(STR, STRB, IMM_OFF, 0, 0, 1)
ARM_MEM_OP_SINGLE(STR, STRB, IMM_OFF_POSTIND, 1, 1, 1)

ARM_MEM_OP_SINGLE(LDR, LDRB, IMM_OFF_PREIND, 0, 1, 1)
ARM_MEM_OP_SINGLE(LDR, LDRB, IMM_OFF, 0, 0, 1)
ARM_MEM_OP_SINGLE(LDR, LDRB, IMM_OFF_POSTIND, 1, 1, 1)

#define ARM_MEM_HALF_OP_DEF2(kind, T, P, post, writeback, size, sign) \
	ARM_##kind##_IMPL(T##_##P##M_REG_OFF, auto rm = regman.get(REG_POS(instr, 0));, c.SUB(addr, rn, rm);, post, writeback, size, sign) \
	ARM_##kind##_IMPL(T##_##P##P_REG_OFF, auto rm = regman.get(REG_POS(instr, 0));, c.ADD(addr, rn, rm);, post, writeback, size, sign) \
	ARM_##kind##_IMPL(T##_##P##M_IMM_OFF, auto imm = (instr&0xf)|((instr>>4)&0xf0);, c.SUB(addr, rn, imm);, post, writeback, size, sign) \
	ARM_##kind##_IMPL(T##_##P##P_IMM_OFF, auto imm = (instr&0xf)|((instr>>4)&0xf0);, c.ADD(addr, rn, imm);, post, writeback, size, sign)

#define ARM_MEM_HALF_OP_DEF(kind, T, size, sign) \
   ARM_MEM_HALF_OP_DEF2(kind, T, POS_INDE_, 1, 1, size, sign); \
   ARM_MEM_HALF_OP_DEF2(kind, T, , 0, 0, size, sign); \
   ARM_MEM_HALF_OP_DEF2(kind, T, PRE_INDE_, 0, 1, size, sign)

ARM_MEM_HALF_OP_DEF(STR, STRH, 2, 0);
ARM_MEM_HALF_OP_DEF(LDR, LDRH, 2, 0);
ARM_MEM_HALF_OP_DEF(LDR, LDRSB, 1, 1);
ARM_MEM_HALF_OP_DEF(LDR, LDRSH, 2, 1);

#define ARM_LDM_STM_INC(before, body) \
	for (int i = 0; i < 16; i++) \
	{ \
		if (instr&BIT(i)) \
		{ \
			if (before) \
				c.ADD(addr, addr, 4); \
			body \
			if (!before) \
				c.ADD(addr, addr, 4); \
		} \
	}
#define ARM_LDM_STM_DEC(before, body) \
	for (int i = 15; i >= 0; i--) \
	{ \
		if (instr&BIT(i)) \
		{ \
			if (before) \
				c.SUB(addr, addr, 4); \
			body \
			if (!before) \
				c.SUB(addr, addr, 4); \
		} \
	}

#define ARM_IMPL_LDM(name, loop, before, writeback) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		if (instr&BIT(15)||!(instr&0xffff)) return 0; \
		auto rn = regman.get(REG_POS(instr,16),writeback); \
		ARM_IMPL_BRANCH( \
			auto rn_in_rlist = instr&BIT(REG_POS(instr,16)); \
			auto addr = regman.new_temp(); \
			c.MOV(addr, rn); \
			auto cycles = regman.new_temp(); \
			c.MOVZ(cycles, 0); \
			loop(before, \
				c.MOV(W0, addr); \
				ARM_MEM_F_IDX(0,4) \
				regman.call(mem_funcs[func_idx]); \
				auto ldr_cycles = regman.new_temp(); \
				c.UBFX(EncodeRegTo64(ldr_cycles), X0, 32, 32); \
				c.ADD(cycles, cycles, ldr_cycles); \
				regman.free_temp(ldr_cycles); \
				if (regman.is_loaded(i)) \
					c.MOV(regman.get(i, true), W0); \
				else \
					c.STR(INDEX_UNSIGNED, W0, RCPU, offsetof(armcpu_t, R) + 4 * i); \
			) \
			if (INSTR_CTX_PROC(ctx) == 0) \
			{ \
				auto res = regman.new_temp(); \
				c.MOVZ(res, 2); \
				c.CMP(cycles, res); \
				c.CSEL(res, cycles, res, CC_GT); \
				c.SUB(cycles, res, 1); \
				regman.free_temp(res); \
			} \
			else \
				c.ADD(cycles, cycles, 1); /* 2 ALU cycles, though one is already there */ \
			c.ADD(Rtotal_cycles, Rtotal_cycles, cycles); \
			regman.free_temp(cycles); \
			if (writeback) \
			{ \
				auto bitlist = (~((2 << REG_POS(instr,16))-1)) & 0xFFFF; \
				if (rn_in_rlist) \
				{ \
					if (bitlist & instr) \
						c.MOV(rn, addr); \
				} \
				else \
					c.MOV(rn, addr); \
			} \
			if (addr != rn) \
				regman.free_temp(addr); \
		, NZCV_TRASH_ALL) \
		return 1; \
	}
#define ARM_IMPL_STM(name, loop, before, writeback) \
	/*ArmOpCompiler ARM_OP_##name = nullptr;*/ \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		if (!(instr&0xffff)) return 0; \
		auto rn = regman.get(REG_POS(instr,16),writeback); \
		ARM_IMPL_BRANCH( \
			auto addr = regman.new_temp(); \
			c.MOV(addr, rn); \
			auto cycles = regman.new_temp(); \
			c.MOVZ(cycles, 0); \
			loop(before, \
				c.MOV(W0, addr); \
				ARM_MEM_F_IDX(1,4) \
				if (regman.is_loaded(i)) \
					c.MOV(W1, regman.get(i)); \
				else \
					c.LDR(INDEX_UNSIGNED, W1, RCPU, offsetof(armcpu_t, R) + 4 * i); \
				regman.call(mem_funcs[func_idx]); \
				c.ADD(cycles, cycles, W0); \
			) \
			if (INSTR_CTX_PROC(ctx) == 0) \
			{ \
				auto res = regman.new_temp(); \
				c.MOVZ(res, 1); \
				c.CMP(cycles, res); \
				c.CSEL(res, cycles, res, CC_GT); \
				c.SUB(cycles, res, 1); \
				regman.free_temp(res); \
			} \
			/* 1 ALU cycle == one constant cycle */ \
			c.ADD(Rtotal_cycles, Rtotal_cycles, cycles); \
			regman.free_temp(cycles); \
			if (writeback) \
				c.MOV(rn, addr); \
			if (addr != rn) \
				regman.free_temp(addr); \
		, NZCV_TRASH_ALL) \
		return 1; \
	}

ARM_IMPL_LDM(LDMIA, ARM_LDM_STM_INC, 0, 0)
ARM_IMPL_LDM(LDMIB, ARM_LDM_STM_INC, 1, 0)
ARM_IMPL_LDM(LDMDA, ARM_LDM_STM_DEC, 0, 0)
ARM_IMPL_LDM(LDMDB, ARM_LDM_STM_DEC, 1, 0)

ARM_IMPL_LDM(LDMIA_W, ARM_LDM_STM_INC, 0, 1)
ARM_IMPL_LDM(LDMIB_W, ARM_LDM_STM_INC, 1, 1)
ARM_IMPL_LDM(LDMDA_W, ARM_LDM_STM_DEC, 0, 1)
ARM_IMPL_LDM(LDMDB_W, ARM_LDM_STM_DEC, 1, 1)

ARM_IMPL_STM(STMIA, ARM_LDM_STM_INC, 0, 0)
ARM_IMPL_STM(STMIB, ARM_LDM_STM_INC, 1, 0)
ARM_IMPL_STM(STMDA, ARM_LDM_STM_DEC, 0, 0)
ARM_IMPL_STM(STMDB, ARM_LDM_STM_DEC, 1, 0)

ARM_IMPL_STM(STMIA_W, ARM_LDM_STM_INC, 0, 1)
ARM_IMPL_STM(STMIB_W, ARM_LDM_STM_INC, 1, 1)
ARM_IMPL_STM(STMDA_W, ARM_LDM_STM_DEC, 0, 1)
ARM_IMPL_STM(STMDB_W, ARM_LDM_STM_DEC, 1, 1)

#define SIGNEXTEND_24(i) (((s32)i<<8)>>8)

#define ARM_IMPL_B_BL(name, link_cond, thumb_off) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		u32 offset = SIGNEXTEND_24(instr); \
		auto r15 = regman.get(15, true); \
		auto is_bx = CONDITION(instr) == 0xf; \
		auto lr = link_cond ? regman.get(14, true) : WZR; \
		auto next_instr = regman.new_temp(); \
		c.MOVI2R(next_instr, pc + 4); \
		c.ADD(r15, next_instr, 4); \
		ARM_IMPL_BRANCH( \
			if (link_cond) \
				c.MOV(lr, next_instr); \
			\
			auto target = (pc + 8 + 4 * offset + (is_bx ? thumb_off : 0))&(0xFFFFFFFC|(is_bx<<1)); \
			if (!branches) \
				constant_branch = target; \
			c.MOVI2R(r15, target); \
			c.MOV(next_instr, r15); \
			\
			if (is_bx) \
			{ \
				auto tmp = regman.new_temp(); \
				JIT_IMM_PRINT("blx imm\n"); \
				c.MOVZ(tmp, 1); \
				c.BFI(RCPSR, tmp, 5, 1); \
				regman.free_temp(tmp); \
				cpsr_dirty = true; \
			} \
			\
		, NZCV_KEPT) \
		c.STR(INDEX_UNSIGNED, next_instr, RCPU, offsetof(armcpu_t, next_instruction)); \
		regman.free_temp(next_instr); \
		return 1; \
	}

ARM_IMPL_B_BL(B, is_bx, 0)
ARM_IMPL_B_BL(BL, true, 2)

#define ARM_IMPL_BX_BLX(name, link_cond) \
	u32 ARM_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto r15 = regman.get(15, true); \
		auto lr = link_cond ? regman.get(14, true) : WZR; \
		auto rn = regman.get(REG_POS(instr,0)); \
		auto next_instr = regman.new_temp(); \
		c.MOVI2R(next_instr, pc + 4); \
		c.ADD(r15, next_instr, 4); \
		ARM_IMPL_BRANCH( \
			auto mask = regman.new_temp(); \
			c.MOVI2R(mask, 0xFFFFFFFC); \
			c.BFI(mask, rn, 1, 1); \
			\
			c.BFI(RCPSR, rn, 5, 1); \
			cpsr_dirty = true; \
			\
			c.AND(r15, rn, mask); \
			if (link_cond) \
				c.MOV(lr, next_instr); /* lr might be the same register as rn */ \
			c.MOV(next_instr, r15); \
			regman.free_temp(mask); \
		, NZCV_KEPT) \
		c.STR(INDEX_UNSIGNED, next_instr, RCPU, offsetof(armcpu_t, next_instruction)); \
		regman.free_temp(next_instr); \
		return 1; \
	}

ARM_IMPL_BX_BLX(BX, 0)
ARM_IMPL_BX_BLX(BLX_REG, 1)

#define ARM_OP_LDRD_STRD_POST_INDEX 0
#define ARM_OP_LDRD_STRD_OFFSET_PRE_INDEX 0
#define ARM_OP_MRS_CPSR 0
#define ARM_OP_SWP 0
#define ARM_OP_MSR_CPSR 0
#define ARM_OP_BKPT 0
#define ARM_OP_MRS_SPSR 0
#define ARM_OP_SWPB 0
#define ARM_OP_MSR_SPSR 0
#define ARM_OP_STREX 0
#define ARM_OP_LDREX 0
#define ARM_OP_MSR_CPSR_IMM_VAL 0
#define ARM_OP_MSR_SPSR_IMM_VAL 0
#define ARM_OP_STMDA2 0
#define ARM_OP_LDMDA2 0
#define ARM_OP_STMDA2_W 0
#define ARM_OP_LDMDA2_W 0
#define ARM_OP_STMIA2 0
#define ARM_OP_LDMIA2 0
#define ARM_OP_STMIA2_W 0
#define ARM_OP_LDMIA2_W 0
#define ARM_OP_STMDB2 0
#define ARM_OP_LDMDB2 0
#define ARM_OP_STMDB2_W 0
#define ARM_OP_LDMDB2_W 0
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

// thumb register
#define REG_T(i, n) (((i)>>n)&0x7)

#define THUMB_FLAGS_OP(n,z,c,v) ((((n)&1)<<7)|(((z)&1)<<6)|(((c)&1)<<5)|(((v)&1)<<4))

#define THUMB_COPY_NZ \
	/*auto nzcv = regman.new_temp(); \
	c.MRS(EncodeRegTo64(nzcv), FIELD_NZCV); \
	c.UBFX(nzcv, nzcv, 30, 2); \
	c.BFI(RCPSR, nzcv, 30, 2); \
	regman.free_temp(nzcv);*/ \
	cpsr_dirty = true;

#define THUMB_IMPL_SHIFT(name, skind, zero) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(1,1,!zero,0); \
	/*ArmOpCompiler THUMB_OP_##name = nullptr;*/ \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_NZ_USEFUL); \
		u32 shift = (instr>>6)&0x1f; \
		auto rd = regman.get(REG_T(instr, 0), true); \
		auto rm = regman.get(REG_T(instr, 3)); /* actually rs */ \
		ARM_SETUP_C_SHIFT_IMM(skind) \
		if (skind != ST_LSL && zero) \
			c.MOV(rd, skind==ST_LSR?WZR:rm, ArithOption(rd, skind, skind==ST_ASR?31:0)); \
		else \
			c.MOV(rd, rm, ArithOption(rd, skind, shift)); \
		c.TST(rd, rd); \
		ARM_ALU_COPY_NZC \
		return 1; \
	}
THUMB_IMPL_SHIFT(LSL, ST_LSL, 0)
THUMB_IMPL_SHIFT(LSL_0, ST_LSL, 1)
THUMB_IMPL_SHIFT(LSR, ST_LSR, 0)
THUMB_IMPL_SHIFT(LSR_0, ST_LSR, 1)
THUMB_IMPL_SHIFT(ASR, ST_ASR, 0)
THUMB_IMPL_SHIFT(ASR_0, ST_ASR, 1)

#define THUMB_IMPL_ADDSUB_REGIMM(name, a64inst, reg) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(1,1,1,1); \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_NZCV_USEFUL); \
		auto rd = regman.get(REG_T(instr, 0), true); \
		auto rs = regman.get(REG_T(instr, 3)); \
		auto rn = reg?regman.get(REG_T(instr, 6)):WZR; \
		auto imm = (instr>>6)&0x7; \
		if (reg) \
			c.a64inst(rd, rs, rn); \
		else \
			c.a64inst(rd, rs, imm); \
		ARM_ALU_COPY_NZCV \
		return 1; \
	}
THUMB_IMPL_ADDSUB_REGIMM(ADD_REG, ADDS, 1)
THUMB_IMPL_ADDSUB_REGIMM(SUB_REG, SUBS, 1)
THUMB_IMPL_ADDSUB_REGIMM(ADD_IMM3, ADDS, 0)
THUMB_IMPL_ADDSUB_REGIMM(SUB_IMM3, SUBS, 0)

#define THUMB_IMPL_ADD_SUB_IMM8(name, a64inst, cmp) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(1,1,1,1); \
	/*ArmOpCompiler THUMB_OP_##name = nullptr;*/ \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_NZCV_USEFUL); \
		auto rd = regman.get(REG_T(instr, 8), !cmp); \
		c.a64inst(cmp?WZR:rd, rd, instr&0xff); \
		ARM_ALU_COPY_NZCV \
		return 1; \
	}
THUMB_IMPL_ADD_SUB_IMM8(CMP_IMM8, SUBS, 1)
THUMB_IMPL_ADD_SUB_IMM8(ADD_IMM8, ADDS, 0)
THUMB_IMPL_ADD_SUB_IMM8(SUB_IMM8, SUBS, 0)

u8 THUMB_FLAGS_OP_MOV_IMM8 = THUMB_FLAGS_OP(1,1,0,0);
u32 THUMB_OP_MOV_IMM8(u32 pc, u32 instr, u32 ctx)
{
	update_nzcv(NZCV_NZ_USEFUL);
	auto rd = regman.get(REG_T(instr, 8), true);
	c.MOVI2R(rd, instr&0xff);
	c.TST(rd,rd);
	THUMB_COPY_NZ
	return 1;
}

#define THUMB_OP_INTERPRET       0
#define THUMB_OP_UND_THUMB       THUMB_OP_INTERPRET
#define THUMB_FLAGS_OP_UND_THUMB       THUMB_FLAGS_OP(1,1,1,1)
#define THUMB_FLAGS_OP_DONTKNOW	THUMB_FLAGS_OP(0,0,0,0)

#define THUMB_OP_LOGIC(name, a64inst, copy_flags) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(1,1,0,0); \
	/*ArmOpCompiler THUMB_OP_##name = nullptr;*/ \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_NZ_USEFUL); \
		auto rd = regman.get(REG_T(instr,0),true); \
		auto rs = regman.get(REG_T(instr,3)); \
		c.a64inst(rd, rd, rs); \
		copy_flags \
		return 1; \
	}
THUMB_OP_LOGIC(AND, ANDS, THUMB_COPY_NZ)
THUMB_OP_LOGIC(EOR, EOR, ARM_ALU_TST_RD THUMB_COPY_NZ)
THUMB_OP_LOGIC(ORR, ORR, ARM_ALU_TST_RD THUMB_COPY_NZ)
THUMB_OP_LOGIC(BIC, BIC, ARM_ALU_TST_RD THUMB_COPY_NZ)

#define THUMB_OP_SHIFT_REG(name, skind) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(1,1,1,0); \
	/*ArmOpCompiler THUMB_OP_##name = nullptr;*/ \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_NZ_USEFUL); \
		auto rm = regman.get(REG_T(instr,0),true); /* actually rd */ \
		auto rs = regman.get(REG_T(instr,3)); \
		ARM_SETUP_SHIFT_REG_S(skind) \
		c.ANDS(rm, op2, op2); \
		regman.free_temp(op2); \
		ARM_ALU_COPY_NZC \
		return 1; \
	}
THUMB_OP_SHIFT_REG(LSL_REG,ST_LSL)
THUMB_OP_SHIFT_REG(LSR_REG,ST_LSR)
THUMB_OP_SHIFT_REG(ASR_REG,ST_ASR)
THUMB_OP_SHIFT_REG(ROR_REG,ST_ROR)

#define THUMB_IMPL_ALU(name, a64inst,compare, copy_flags, flags, arithmetic) \
	u8 THUMB_FLAGS_OP_##name = flags; \
	/*ArmOpCompiler THUMB_OP_##name = nullptr;*/ \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(arithmetic ? NZCV_NZCV_USEFUL : NZCV_NZ_USEFUL); \
		auto rd = regman.get(REG_T(instr,0),!compare); \
		auto rs = regman.get(REG_T(instr,3)); \
		c.a64inst(rd, rs); \
		copy_flags \
		return 1; \
	}
THUMB_IMPL_ALU(TST, TST, 1, THUMB_COPY_NZ, THUMB_FLAGS_OP(1,1,0,0), 0)
THUMB_IMPL_ALU(CMP, CMP, 1, ARM_ALU_COPY_NZCV, THUMB_FLAGS_OP(1,1,1,1), 1)
THUMB_IMPL_ALU(CMN, CMN, 1, ARM_ALU_COPY_NZCV, THUMB_FLAGS_OP(1,1,1,1), 1)
THUMB_IMPL_ALU(MVN, MVN, 0, ARM_ALU_TST_RD THUMB_COPY_NZ, THUMB_FLAGS_OP(1,1,0,0), 0)
u8 THUMB_FLAGS_OP_NEG = THUMB_FLAGS_OP(1,1,1,1);
u32 THUMB_OP_NEG(u32 pc, u32 instr, u32 ctx) 
{
	update_nzcv(NZCV_NZCV_USEFUL);
	auto rd = regman.get(REG_T(instr,0),true);
	auto rs = regman.get(REG_T(instr,3));
	c.SUBS(rd, WZR, rs);
	ARM_ALU_COPY_NZCV
	return 1; 
}

#define THUMB_OP_ALU nullptr
#define THUMB_OP_ADC_REG         THUMB_OP_ALU
#define THUMB_OP_SBC_REG         THUMB_OP_ALU
#define THUMB_OP_MUL_REG         THUMB_OP_INTERPRET

#define THUMB_FLAGS_OP_ADC_REG         THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_SBC_REG         THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_MUL_REG         THUMB_FLAGS_OP_DONTKNOW

#define THUMB_IMPL_LDR(name, calc_addr, size, sign) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(0,0,0,0); \
	ArmOpCompiler THUMB_OP_##name = nullptr;

#define THUMB_MEM_REG_OFF(ldr) \
	auto rb = regman.get(REG_T(instr,3)); \
	auto rd = regman.get(REG_T(instr,0),ldr); \
	auto ro = regman.get(REG_T(instr,6));
#define THUMB_LDR_REG_OFF THUMB_MEM_REG_OFF(true)
#define THUMB_STR_REG_OFF THUMB_MEM_REG_OFF(false)

#define THUMB_MEM_IMM_OFF(ldr) \
	auto rb = regman.get(REG_T(instr,3)); \
	auto rd = regman.get(REG_T(instr,0),ldr); \
	auto imm_unscaled = (instr>>6)&0x1f;
#define THUMB_LDR_IMM_OFF THUMB_MEM_IMM_OFF(true)
#define THUMB_STR_IMM_OFF THUMB_MEM_IMM_OFF(false)

#define THUMB_IMPL_LDRB(name, calc_addr, size, sign) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(0,0,0,0); \
	/*ArmOpCompiler THUMB_OP_##name = nullptr;*/ \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_TRASH_ALL); \
		calc_addr \
		ARM_MEM_F_IDX(0, size) \
		regman.call(mem_funcs[func_idx]); \
		auto cycles = regman.new_temp(); \
		c.UBFX(EncodeRegTo64(cycles), X0, 32, 32); \
		c.ADD(Rtotal_cycles, Rtotal_cycles, cycles); \
		c.SUB(Rtotal_cycles, Rtotal_cycles, 1); \
		regman.free_temp(cycles); \
		ARM_MEM_MOVXT(rd, W0, size, sign) \
		return 1; \
	}

#define THUMB_IMPL_STR(name, calc_addr, size, sign) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(0,0,0,0); \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		update_nzcv(NZCV_TRASH_ALL); \
		calc_addr \
		ARM_MEM_MOVXT(W1, rd, size, sign) \
		ARM_MEM_F_IDX(1, size) \
		regman.call(mem_funcs[func_idx]); \
		c.ADD(Rtotal_cycles, Rtotal_cycles, W0); \
		c.SUB(Rtotal_cycles, Rtotal_cycles, 1); \
		return 1; \
	}

THUMB_IMPL_LDRB(LDR_SPREL, 
	auto rd = regman.get(REG_T(instr,8),true); 
	c.ADD(W0,regman.get(13),(instr&0xff)*4);, 4, 0)
THUMB_IMPL_LDRB(LDR_PCREL,
	auto rd = regman.get(REG_T(instr,8),true);
	c.MOVI2R(W0,((pc+4)&0xFFFFFFFC)+(instr&0xff)*4);,4, 0)
THUMB_IMPL_STR(STR_SPREL,
	auto rd = regman.get(REG_T(instr,8));
	c.ADD(W0,regman.get(13),(instr&0xff)*4);, 4, 0)

u8 THUMB_FLAGS_OP_ADD_2PC = THUMB_FLAGS_OP(0,0,0,0);
u32 THUMB_OP_ADD_2PC(u32 pc, u32 instr, u32 ctx)
{
	auto rd = regman.get(REG_T(instr,8),true);
	c.MOVI2R(rd, ((pc+4)&0xFFFFFFFC)+(instr&0xff)*4);
	return 1;
}
u8 THUMB_FLAGS_OP_ADD_2SP = THUMB_FLAGS_OP(0,0,0,0);
u32 THUMB_OP_ADD_2SP(u32 pc, u32 instr, u32 ctx)
{
	auto rd = regman.get(REG_T(instr,8),true);
	c.ADD(rd, regman.get(13), (instr&0xff)*4);
	return 1;
}

#define SIGNEEXT_IMM11(i)	(((i)&0x7FF) | (BIT10(i) * 0xFFFFF800))

u8 THUMB_FLAGS_OP_B_UNCOND = THUMB_FLAGS_OP(0,0,0,0);
u32 THUMB_OP_B_UNCOND(u32 pc, u32 instr, u32 ctx)
{
	auto r15 = regman.get(15, true);
	auto offset = SIGNEEXT_IMM11(instr) * 2;
	constant_branch = pc + 4 + offset;
	c.MOVI2R(r15, constant_branch);
	c.STR(INDEX_UNSIGNED, r15, RCPU, offsetof(armcpu_t, next_instruction));
	return 1;
}

u8 THUMB_FLAGS_OP_B_COND = THUMB_FLAGS_OP(0,0,0,0);
u32 THUMB_OP_B_COND(u32 pc, u32 instr, u32 ctx)
{
	auto offset = (s8)(instr&0xFF);

	auto r15 = regman.get(15, true);
	auto next_instr = regman.new_temp();
	c.MOVI2R(next_instr, pc + 2);
	c.ADD(r15, next_instr, 2);

	/*c._MSR(FIELD_NZCV, EncodeRegTo64(RCPSR));*/
	auto __skip = /*c.B((CCFlags)(((instr>>8)&0xf)^1))*/branch_on_nzcv((instr>>8)&0xf, nzcv_location);

	if (offset < 0)
		c.SUB(next_instr, r15, (u32)(-offset) * 2);
	else
		c.ADD(next_instr, r15, (u32)offset * 2);
	c.MOV(r15, next_instr);

	c.SetJumpTarget(__skip);

	c.STR(INDEX_UNSIGNED, next_instr, RCPU, offsetof(armcpu_t, next_instruction));

	regman.free_temp(next_instr);
	return 1;
}

#define THUMB_OP_BL_LONG nullptr
#define THUMB_FLAGS_OP_BL_LONG THUMB_FLAGS_OP_DONTKNOW

#define THUMB_OP_SPE nullptr
#define THUMB_OP_ADD_SPE         THUMB_OP_SPE
#define THUMB_OP_CMP_SPE         THUMB_OP_SPE
#define THUMB_OP_MOV_SPE         THUMB_OP_SPE

#define THUMB_FLAGS_OP_ADD_SPE         THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_CMP_SPE         THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_MOV_SPE         THUMB_FLAGS_OP_DONTKNOW

#define THUMB_IMPL_ADJUST_SP(name, neg) \
	u8 THUMB_FLAGS_OP_##name = THUMB_FLAGS_OP(0,0,0,0); \
	u32 THUMB_OP_##name(u32 pc, u32 instr, u32 ctx) \
	{ \
		auto sp = regman.get(13, true); \
		auto imm = (instr&0x7F)*4; \
		if (neg) \
			c.SUB(sp, sp, imm); \
		else \
			c.ADD(sp, sp, imm); \
		return 1; \
	}
THUMB_IMPL_ADJUST_SP(ADJUST_P_SP, 0)
THUMB_IMPL_ADJUST_SP(ADJUST_M_SP, 1)

THUMB_IMPL_LDRB(LDRB_REG_OFF,THUMB_LDR_REG_OFF c.ADD(W0,rb,ro);,1,0)
THUMB_IMPL_LDRB(LDRH_REG_OFF,THUMB_LDR_REG_OFF c.ADD(W0,rb,ro);,2,0)
THUMB_IMPL_LDRB(LDR_REG_OFF,THUMB_LDR_REG_OFF c.ADD(W0,rb,ro);,4,0)

THUMB_IMPL_STR(STRB_REG_OFF,THUMB_STR_REG_OFF c.ADD(W0,rb,ro);,1,0)
THUMB_IMPL_STR(STRH_REG_OFF,THUMB_STR_REG_OFF c.ADD(W0,rb,ro);,2,0)
THUMB_IMPL_STR(STR_REG_OFF,THUMB_STR_REG_OFF c.ADD(W0,rb,ro);,4,0)

THUMB_IMPL_LDRB(LDRB_IMM_OFF,THUMB_LDR_IMM_OFF c.ADD(W0,rb,imm_unscaled);,1,0)
THUMB_IMPL_LDRB(LDRH_IMM_OFF,THUMB_LDR_IMM_OFF c.ADD(W0,rb,imm_unscaled*2);,2,0)
THUMB_IMPL_LDRB(LDR_IMM_OFF,THUMB_LDR_IMM_OFF c.ADD(W0,rb,imm_unscaled*4);,4,0)

THUMB_IMPL_STR(STRB_IMM_OFF,THUMB_STR_IMM_OFF c.ADD(W0,rb,imm_unscaled);,1,0)
THUMB_IMPL_STR(STRH_IMM_OFF,THUMB_STR_IMM_OFF c.ADD(W0,rb,imm_unscaled*2);,2,0)
THUMB_IMPL_STR(STR_IMM_OFF,THUMB_STR_IMM_OFF c.ADD(W0,rb,imm_unscaled*4);,4,0)

THUMB_IMPL_LDRB(LDRSB_REG_OFF,THUMB_LDR_REG_OFF c.ADD(W0,rb,ro);,1,1)
THUMB_IMPL_LDRB(LDRSH_REG_OFF,THUMB_LDR_REG_OFF c.ADD(W0,rb,ro);,2,1)

#define SIGNEXTEND_11(i) (((s32)i<<21)>>21)

u32 THUMB_OP_BL_10(u32 pc, u32 instr, u32 ctx)
{
	auto r14 = regman.get(14, true);
	constant_branch = pc + 4 + (SIGNEXTEND_11(instr)<<12);
	c.MOVI2R(r14, constant_branch);
	return 1;
}

u32 THUMB_OP_BL_11(u32 pc, u32 instr, u32 ctx)
{
	auto r15 = regman.get(15, true);
	auto r14 = regman.get(14, true);
	constant_branch += (instr&0x7FF)<<1;
	c.MOVI2R(r15, constant_branch);
	c.MOVI2R(r14, (pc + 2) | 1);
	c.STR(INDEX_UNSIGNED, r15, RCPU, offsetof(armcpu_t, next_instruction));
	return 1;
}

#define THUMB_OP_BX_BLX_THUMB nullptr
#define THUMB_OP_BL_LONG nullptr
#define THUMB_OP_BX_THUMB        THUMB_OP_BX_BLX_THUMB
#define THUMB_OP_BLX_THUMB       THUMB_OP_BX_BLX_THUMB
//#define THUMB_OP_BL_10           THUMB_OP_BL_LONG
//#define THUMB_OP_BL_11           THUMB_OP_BL_LONG
#define THUMB_OP_BLX             THUMB_OP_BL_LONG

#define THUMB_FLAGS_OP_BX_THUMB        THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_BLX_THUMB       THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_BL_10           THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_BL_11           THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_BLX             THUMB_FLAGS_OP_DONTKNOW

// UNDEFINED OPS
#define THUMB_OP_PUSH            THUMB_OP_INTERPRET
#define THUMB_OP_PUSH_LR         THUMB_OP_INTERPRET
#define THUMB_OP_POP             THUMB_OP_INTERPRET
#define THUMB_OP_POP_PC          THUMB_OP_INTERPRET
#define THUMB_OP_BKPT_THUMB      THUMB_OP_INTERPRET
#define THUMB_OP_STMIA_THUMB     THUMB_OP_INTERPRET
#define THUMB_OP_LDMIA_THUMB     THUMB_OP_INTERPRET
#define THUMB_OP_SWI_THUMB       THUMB_OP_INTERPRET

#define THUMB_FLAGS_OP_PUSH            THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_PUSH_LR         THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_POP             THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_POP_PC          THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_BKPT_THUMB      THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_STMIA_THUMB     THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_LDMIA_THUMB     THUMB_FLAGS_OP_DONTKNOW
#define THUMB_FLAGS_OP_SWI_THUMB       THUMB_FLAGS_OP_DONTKNOW

// Thumb
static const ArmOpCompiler thumb_instruction_compilers[1024] = {
#define TABDECL(x) THUMB_##x
#include "thumb_tabdef.inc"
#undef TABDECL
};

static const u8 thumb_instruction_set_flag[1024] =
{
#define TABDECL(x) THUMB_FLAGS_##x
#include "thumb_tabdef.inc"
#undef TABDECL
};


static u32 instr_attributes(bool thumb, u32 opcode)
{
	return thumb ? thumb_attributes[opcode>>6]
		 : instruction_attributes[INSTRUCTION_INDEX(opcode)];
}

static bool instr_uses_r15(bool thumb, u32 opcode)
{
	u32 x = instr_attributes(thumb, opcode);
	if(thumb)
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

static bool instr_is_branch(bool thumb, u32 opcode)
{
	u32 x = instr_attributes(thumb, opcode);
	
	if(thumb)
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

template <int PROCNUM>
static int instr_cycles(bool thumb, u32 opcode)
{
	u32 x = instr_attributes(thumb, opcode);
	u32 c = (x & INSTR_CYCLES_MASK);
	if(c == INSTR_CYCLES_VARIABLE)
	{
		if ((x & BRANCH_SWI) && !ARMPROC.swi_tab)
			return 3;
		
		return 0;
	}
	if(instr_is_branch(thumb, opcode) && !(instr_attributes(thumb, opcode) & (BRANCH_ALWAYS|BRANCH_LDM)))
		c += 2;
	return c;
}

static const char *disassemble(bool thumb, u32 opcode)
{
	if(thumb)
		return thumb_instruction_names[opcode>>6];
	static char str[100];
	strcpy(str, arm_instruction_names[INSTRUCTION_INDEX(opcode)]);
	static const char *conds[16] = {"EQ","NE","CS","CC","MI","PL","VS","VC","HI","LS","GE","LT","GT","LE","AL","NV"};
	if(instr_is_conditional(thumb, opcode))
	{
		strcat(str, ".");
		strcat(str, conds[CONDITION(opcode)]);
	}
	return str;
}

/*

    Kriegen ihr eigenes Register, dauerhaft gemappt:
        ARMPROC -> X28
        Cyclenzähler -> W27
        CPSR -> W26

        W19-W25, also 7 Register bleiben übrig


    - Branches werden innerhalb der Compilerfunktionen gehandhabt
    - Müssen in Variablen gespeichert werden: 
        auto rd = map_reg(opcode, 12);
        …
    - Wird auf begrenzte Zahl von Registern gemappt
*/

static u32 saved_regs[16];
static u32 saved_cpsr;
static void save_regs(armcpu_t* cpu)
{
	for (int i = 0; i < 15; i++)
		saved_regs[i] = cpu->R[i];
	saved_cpsr = cpu->CPSR.val;
}
#define SWAP(a, b, c) do      \
	              {       \
                         c=a; \
                         a=b; \
                         b=c; \
		      }       \
                      while(0)
static void restore_regs(armcpu_t* cpu)
{
	u32 tmp;
	for (int i = 0; i < 15; i++)
		SWAP(cpu->R[i], saved_regs[i], tmp);
	SWAP(cpu->CPSR.val, saved_cpsr, tmp);
}
static void cmp_regs(armcpu_t* cpu, u32 opcode)
{
	auto mismatch = false;
	for (int i = 0; i < 15; i++)
		mismatch |= saved_regs[i] != cpu->R[i];
	mismatch |= cpu->CPSR.val != saved_cpsr;
	if (mismatch)
	{
		printf("conflict at instr %x\n", opcode);
		printf("expected:  got: \n");
		for (int i = 0; i < 15;i++)
			printf("r%d %x | %x\n", i, cpu->R[i], saved_regs[i]);
		printf("CPSR: %x | %x\n", cpu->CPSR.val, saved_cpsr);
	}
}

#ifdef JIT_DEBUG
extern bool jit_second_run;
#endif

#include "switch/profiler.h"

template<int PROCNUM>
static ArmOpCompiled compile_basicblock()
{
	//profiler::Section profiler("JIT compilation");

    auto base_addr = ARMPROC.instruct_adr;
    auto thumb = ARMPROC.CPSR.bits.T;
    auto opcode_size = thumb ? 2 : 4;

	u32 constant_cycles = 0;
	u32 opcode = 0;
	u32 addr = base_addr;

	constant_branch = 0;

    if (!JIT_MAPPED(base_addr & 0x0FFFFFFF, PROCNUM))
	{
		printf("JIT: use unmapped memory address %08X\n", base_addr);
		execute = false;
		return nullptr;
	}

    libnx::jitTransitionToWritable(&jit_block);

    auto f = (ArmOpCompiled)(c.GetCodePtr() - jit_rw_addr + jit_rx_addr);
    JIT_COMPILED_FUNC(base_addr, PROCNUM) = (uintptr_t)f;

	BitSet32 stashed_regs({1, 30, (int)DecodeReg(RCPU), (int)RCPSR, (int)Rtotal_cycles, 19, 20, 21, 22, 23, 24, 25});
    c.ABI_PushRegisters(stashed_regs);

	c.MOVP2R(RCPU, &ARMPROC);
	c.MOVZ(Rtotal_cycles, W0);

    regman.reset();
    load_cpsr();

	JIT_IMM_PRINT("load block thumb %d\n", thumb);

	u32 instrs_count = 0;
    for (u32 last_instr = false, i = 0; !last_instr; i++)
    {
		addr = base_addr + opcode_size * i;
        opcode = thumb ? _MMU_read16<PROCNUM, MMU_AT_CODE>(addr) :
			_MMU_read32<PROCNUM, MMU_AT_CODE>(addr);

		//printf("%s\n", disassemble(thumb, opcode));

		auto cycles = instr_cycles<PROCNUM>(thumb, opcode);

		last_instr = (instr_is_branch(thumb, opcode) || i + 1 >= CommonSettings.jit_max_block_size) &&
			!(instr_attributes(thumb, opcode) & MERGE_NEXT);
		instrs_count = i + 1;

		auto compiler = thumb ? thumb_instruction_compilers[opcode>>6] : 
			arm_instruction_compilers[INSTRUCTION_INDEX(opcode)];
#ifdef JIT_DEBUG
		if (!jit_second_run)
			compiler = nullptr;
#endif
		
		if (instr_uses_r15(thumb, opcode))
		{
			auto r15 = regman.get(15);
			c.MOVI2R(r15, addr + opcode_size * 2);
		}

		constant_cycles += instr_is_conditional(thumb, opcode) ? 1 : (cycles == 0 ? 1 : cycles);

		if (compiler && (thumb || !((instr_attributes(thumb, opcode) & BRANCH_POS12) && REG_POS(opcode,12)==15)) && compiler(addr, opcode, INSTR_CTX(cycles, PROCNUM)))
		{
			JIT_IMM_PRINT("%s - compiled\t%x\n", disassemble(thumb, opcode), opcode);

#ifdef JIT_DEBUG
			save_cpsr();
			regman.flush_regs();
			c.MOV(X0, RCPU);
			c.MOVI2R(W1, opcode);
			c.MOVI2R(W2, addr);
			regman.call(check_result);
#endif
		}
		else
		{
			JIT_IMM_PRINT("%s\t\t%x\n", disassemble(thumb, opcode), opcode);
			prefetch_addrs(addr, opcode_size);

			OpFunc f = thumb ? thumb_instructions_set[PROCNUM][opcode>>6] :
				arm_instructions_set[PROCNUM][INSTRUCTION_INDEX(opcode)];

			save_cpsr();
			regman.flush_regs();
			regman.reset();
			if (instr_is_conditional(thumb, opcode))
			{
				c._MSR(FIELD_NZCV, EncodeRegTo64(RCPSR));
				auto skip_matches = c.B((CCFlags)(CONDITION(opcode)^1));
				c.MOVI2R(W0, opcode);
				regman.call(f);
				if (cycles == 0)
				{
					c.SUB(W0, W0, 1);
					c.ADD(Rtotal_cycles, Rtotal_cycles, W0);
				}
				else if (cycles > 0)
					c.ADD(Rtotal_cycles, Rtotal_cycles, cycles - 1);
				c.SetJumpTarget(skip_matches);
			}
			else
			{
				c.MOVI2R(W0, opcode);
				regman.call(f);
				if (cycles == 0)
				{
					c.SUB(W0, W0, 1);
					c.ADD(Rtotal_cycles, Rtotal_cycles, W0);
				}
			}

#ifdef JIT_DEBUG
			c.MOV(X0, RCPU);
			c.MOVI2R(W1, opcode);
			c.MOVI2R(W2, addr);
			regman.call(record_result);
#endif

			load_cpsr();
		}
    }

	if (instr_is_branch(thumb, opcode))
	{
		auto tmp = regman.new_temp();
		c.LDR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, next_instruction));
		c.STR(INDEX_UNSIGNED, tmp, RCPU, offsetof(armcpu_t, instruct_adr));
		regman.free_temp(tmp);
	}
	else
		prefetch_addrs(addr + opcode_size, opcode_size);

	regman.flush_regs();

	save_cpsr();

	JIT_IMM_PRINT("%d\n", constant_cycles);
	c.ADD(W0, Rtotal_cycles, constant_cycles);
    c.ABI_PopRegisters(stashed_regs);

	if (constant_branch != 0)
	{
		c.CMP(W1, CommonSettings.jit_max_block_size);
		c.B(CC_GE);
		c.ADD(W1, W1, instrs_count);

		c.MOVP2R(X4, &JIT_COMPILED_FUNC(constant_branch, PROCNUM));
		c.LDR(INDEX_UNSIGNED, X5, X4, 0);
		c.CMP(X5, 0);
		c.B(CC_NEQ);
	}
	c.RET();

    libnx::jitTransitionToExecutable(&jit_block);

    return f;
}

template<int PROCNUM> u32 arm_jit_compile()
{
    u32 addr = ARMPROC.instruct_adr;
	u32 mask_addr = (addr & 0x07FFFFFE) >> 4;
	if(((recompile_counts[mask_addr >> 1] >> 4*(mask_addr & 1)) & 0xF) > 8)
	{
		ArmOpCompiled f = op_decode[PROCNUM][ARMPROC.CPSR.bits.T];
		JIT_COMPILED_FUNC(addr, PROCNUM) = (uintptr_t)f;
		return f(0, 0);
	}
	recompile_counts[mask_addr >> 1] += 1 << 4*(mask_addr & 1);
	
	if((JIT_BLOCK_SIZE - (c.GetCodePtr() - jit_rw_addr)) / 4 < 1000) {
		printf("JIT block full resetting...\n");
		arm_jit_reset(true);
	}

    auto f = compile_basicblock<PROCNUM>();
	u32 cycles = f ? f(0, 0) : 0;
    return cycles;
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

        if (jit_rx_addr == nullptr)
        {
            libnx::jitCreate(&jit_block, JIT_BLOCK_SIZE);
            jit_rw_addr = (u8*)libnx::jitGetRwAddr(&jit_block);
			jit_rx_addr = (u8*)libnx::jitGetRxAddr(&jit_block);
        }
        c.SetCodePtr(jit_rw_addr);

#ifdef JIT_DEBUG
		if (results == nullptr)
			results = (reg_set*)malloc(1024 * 1024 * 16);
#endif
    }
}

void arm_jit_close()
{
    if (jit_rx_addr != nullptr)
        libnx::jitClose(&jit_block);

#ifdef JIT_DEBUG
	if (results != nullptr)
		free(results);
#endif
}