/* main.c - this file is part of DeSmuME
*
* Copyright (C) 2006,2007 DeSmuME Team
* Copyright (C) 2007 Pascal Giard (evilynux)
* Copyright (C) 2009 Yoshihiro (DsonPSP)
* This file is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2, or (at your option)
* any later version.
*
* This file is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; see the file COPYING.  If not, write to
* the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
* Boston, MA 02111-1307, USA.
*/
#include <stdio.h>


//DIRTY FIX FOR CONFLICTING TYPEDEFS
namespace libnx 
{
	#include <switch.h>
}

#include <malloc.h>

#include "../MMU.h"
#include "../NDSSystem.h"
#include "../debug.h"
#include "../render3D.h"
#include "../rasterize.h"
#include "../saves.h"
#include "../mic.h"
#include "../SPU.h"

/*
#include "config.h"*/
#include "input.h"
#include "menu.h"
#include "sound.h"

#include "profiler.h"

volatile bool execute = FALSE;

static inline uint8_t convert5To8(uint8_t x)
{
	return ((x << 3) | (x >> 2));
}

static inline uint32_t ABGR1555toRGBA8(uint16_t color)
{
	uint32_t pixel = 0x00;
	uint8_t  *dst = (uint8_t*)&pixel; 

	dst[0] = convert5To8((color >> 0)   & 0x1F); //R
	dst[1] = convert5To8((color >> 5)   & 0x1F); //G
	dst[2] = convert5To8((color >> 10)  & 0x1F); //B
	dst[3] = 0xFF;//CONVERT_5_TO_8((color >> 11) & 0x1F); //A

    return pixel;
}

GPU3DInterface *core3DList[] = {
	&gpu3DNull,
	&gpu3DRasterize,
	NULL
};

SoundInterface_struct *SNDCoreList[] = {
	&SNDDummy,
	&SNDSwitch,
	NULL
};

const char * save_type_names[] = {
	"Autodetect",
	"EEPROM 4kbit",
	"EEPROM 64kbit",
	"EEPROM 512kbit",
	"FRAM 256kbit",
	"FLASH 2mbit",
	"FLASH 4mbit",
	NULL
};

int cycles;

static unsigned short keypad;

static bool desmume_cycle()
{
	libnx::hidScanInput();

	u32 keysDown = libnx::hidKeysDown(libnx::CONTROLLER_P1_AUTO);
	u32 keysUp = libnx::hidKeysUp(libnx::CONTROLLER_P1_AUTO);
    process_ctrls_events(&keypad, keysDown, keysUp);

    if(libnx::hidTouchCount())
    {
    	libnx::touchPosition touch;
		hidTouchRead(&touch, 0);

		if(touch.px > 401 && touch.px < 882 && touch.py > 360 && touch.py < 720)
		{

				NDS_setTouchPos((touch.px - 401) / 1.875,(touch.py - 360) / 1.875);
		}
	}

	else if(libnx::hidKeysUp(libnx::CONTROLLER_P1_AUTO) & libnx::KEY_TOUCH)
	{
		NDS_releaseTouch();
	}

	update_keypad(keypad);

	profiler::Section frame("frame");
	
    NDS_exec<false>();

    /*if(UserConfiguration.soundEnabled)*/
    	SPU_Emulate_user();

	return keysDown & libnx::KEY_ZL;
}

int main(int argc, char *argv[])
{
	u32 originalCpuFreq = 0;
	if(R_SUCCEEDED(libnx::pcvInitialize())) {
		printf("overclocking!\n");
		libnx::pcvGetClockRate(libnx::PcvModule::PcvModule_Cpu, &originalCpuFreq);
		printf("current freq %d\n", originalCpuFreq);
		//u32 rate = 1020000000;
		u32 rate = 1224000000;
		//u32 rate = 1785000000;
    	libnx::pcvSetClockRate(libnx::PcvModule::PcvModule_Cpu, rate);
		printf("set frequency to %d Hz\n", rate);
   	}
	else
		printf("failed to initalise pcv\n");


	libnx::gfxInitDefault();
	libnx::consoleInit(NULL);

	char *rom_path = menu_FileBrowser();

	libnx::gfxConfigureResolution(684, 384);

	/* the firmware settings */
    FirmwareConfig fw_config;
    NDS_GetDefaultFirmwareConfig(fw_config);
	fw_config.language = 4;
	NDS_InitFirmwareWithConfig(fw_config);

  	NDS_Init();

	libnx::socketInitializeDefault();
	int nxlinkSocket = libnx::nxlinkStdio();

  	GPU->Change3DRendererByID(RENDERID_SOFTRASTERIZER);
	GPU->SetColorFormat(NDSColorFormat_BGR888_Rev);
	GPU->SetCustomFramebufferSize(GPU_FRAMEBUFFER_NATIVE_WIDTH, GPU_FRAMEBUFFER_NATIVE_HEIGHT);
  	SPU_ChangeSoundCore(SNDCORE_SWITCH, 735 * 4);

	CommonSettings.use_jit = true;
	CommonSettings.jit_max_block_size = 100;
	CommonSettings.loadToMemory = true;

	if (NDS_LoadROM( rom_path, NULL, NULL) < 0) {
		printf("Error loading %s\n", rom_path);
	}

	execute = TRUE;
	u32 width, height;

	u64 frametimes[30];
	u32 frames = 0;

	while(libnx::appletMainLoop()) 
	{
		u64 startTime = libnx::armGetSystemTick();

		/*for (int i = 0; i < UserConfiguration.frameSkip; i++)
		{
			NDS_SkipNextFrame();
			desmume_cycle();
		}*/

		if (desmume_cycle())
			break;

		profiler::frame();

		uint32_t * src = (uint32_t *)GPU->GetDisplayInfo().masterNativeBuffer;
		uint32_t *framebuffer = (uint32_t*)libnx::gfxGetFramebuffer(&width, &height);

		for(int x = 0; x < 256; x++){
    		for(int y = 0; y < (192 * 2); y++){
    			uint32_t offset = libnx::gfxGetFramebufferDisplayOffset(214 + x, y);
        		framebuffer[offset] = src[( y * 256 ) + x]|(0xff<<24);
    		}
		}

		u64 endTime = libnx::armGetSystemTick();
		u64 delta = endTime - startTime;
		frametimes[frames++] = delta;

		if (frames == 30)
		{
			float invFreq = 1.f/(float)libnx::armGetSystemTickFreq();

			u64 frametimeSum = 0;
			for (int i = 0; i < 30; i++)
				frametimeSum += frametimes[i];
			float avg = (float)frametimeSum*invFreq/30.f;
			// yey, statistics!!
			float stdDev = 0.f;
			for (int i = 0; i < 30; i++)
			{
				float diff = (frametimes[i]*invFreq)-avg;
				stdDev += diff * diff;
			}
			stdDev = sqrtf(stdDev/30.f);

			printf("frametime %f(+/- %f)\n", avg*1000.f, stdDev*1000.f);
			frametimeSum = 0;
			frames = 0;
		}

		libnx::gfxFlushBuffers();
		libnx::gfxSwapBuffers();
    }

    NDS_FreeROM();
    NDS_DeInit();

	deinit_switch_sound();

	libnx::socketExit();

    libnx::gfxExit();

	if (originalCpuFreq != 0)
	{
		libnx::pcvSetClockRate(libnx::PcvModule_Cpu, originalCpuFreq);
		libnx::pcvExit();
	}

	return 0;
}