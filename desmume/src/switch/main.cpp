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

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "glad.h"

#include <malloc.h>

#include "../MMU.h"
#include "../NDSSystem.h"
#include "../debug.h"
#include "../render3D.h"
#include "../rasterize.h"
#include "../saves.h"
#include "../mic.h"
#include "../SPU.h"
#include "../OGLRender.h"
#include "../OGLRender_3_2.h"

#include <stdlib.h>

/*
#include "config.h"*/
#include "input.h"
#include "menu.h"
#include "sound.h"

#include "profiler.h"

#define MMX_IMPLEMENTATION
#define MMX_STATIC
#include "mm_vec.h"

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
	&gpu3Dgl_3_2,
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

	//profiler::Section frame("frame");
	
    NDS_exec<false>();

    /*if(UserConfiguration.soundEnabled)*/
    	SPU_Emulate_user();

	return keysDown & libnx::KEY_ZL;
}

struct Vertex {
	float xy[2];
	float uv[2];
};

static const char* vertexShaderSource = R"text(
	#version 330 core
    layout (location = 0) in vec2 inXY;
    layout (location = 1) in vec2 inUV;

	uniform mat4 proj;
	uniform float uvOffset;

	out vec2 uv;
    void main()
    {
		uv = inUV;
		uv.y += uvOffset;
        gl_Position = proj * vec4(inXY, 0.0, 1.0);
    }
)text";
static const char* const fragmentShaderSource = R"text(
    #version 330 core
    in vec2 uv;
    out vec4 fragColor;

	uniform sampler2D inTexture;

    void main()
    {
        fragColor = texture(inTexture, uv);
		fragColor.w = 1.0;
    }
)text";

static const Vertex screenVertices[] = {
	{{0.f, 0.f}, {0.f, 0.f}},
	{{0.f, 192.f}, {0.f, 0.5f}},
	{{256.f, 192.f}, {1.f, 0.5f}},

	{{0.f, 0.f}, {0.f, 0.f}},
	{{256.f, 192.f}, {1.f, 0.5f}},
	{{256.f, 0.f}, {1.f, 0.f}}
};

static bool dummy_retro_gl_init() { return true; }
static void dummy_retro_gl_end() {}
static bool dummy_retro_gl_begin() { return true; }

int main(int argc, char *argv[])
{
	if (argc < 2)
		return 0;

	oglrender_init        = dummy_retro_gl_init;
	oglrender_beginOpenGL = dummy_retro_gl_begin;
	oglrender_endOpenGL = dummy_retro_gl_end;

	OGLLoadEntryPoints_3_2_Func = OGLLoadEntryPoints_3_2;
	OGLCreateRenderer_3_2_Func = OGLCreateRenderer_3_2;

	libnx::socketInitializeDefault();
	int nxlinkSocket = libnx::nxlinkStdio();

	u32 originalCpuFreq = 0;
	if(R_SUCCEEDED(libnx::pcvInitialize())) {
		printf("overclocking!\n");
		libnx::pcvGetClockRate(libnx::PcvModule::PcvModule_Cpu, &originalCpuFreq);
		printf("current freq %d\n", originalCpuFreq);
		u32 rate = 1020000000;
		//u32 rate = 1224000000;
		//u32 rate = 1785000000;
    	libnx::pcvSetClockRate(libnx::PcvModule::PcvModule_Cpu, rate);
		printf("set frequency to %d Hz\n", rate);
   	}
	else
		printf("failed to initalise pcv\n");

	libnx::NWindow* win = libnx::nwindowGetDefault();

	/*setenv("EGL_LOG_LEVEL", "debug", 1);
    setenv("MESA_VERBOSE", "all", 1);
	setenv("NOUVEAU_MESA_DEBUG", "1", 1);*/

	EGLDisplay egl_disp = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	eglInitialize(egl_disp, nullptr, nullptr);

	if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE)
		printf("couldn't bind api %d\n", eglGetError());

	EGLConfig config;
	EGLint num_configs;
	static const EGLint framebufferAttributeList[] =
    {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE,     8,
        EGL_GREEN_SIZE,   8,
        EGL_BLUE_SIZE,    8,
        EGL_ALPHA_SIZE,   8,
        EGL_DEPTH_SIZE,   24,
        EGL_STENCIL_SIZE, 8,
        EGL_NONE
	};
	eglChooseConfig(egl_disp, framebufferAttributeList, &config, 1, &num_configs);
	assert(num_configs);

	EGLSurface egl_surface = eglCreateWindowSurface(egl_disp, config, win, nullptr);
	assert(egl_surface);

	static const EGLint contextAttributeList[] =
    {
        EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT_KHR,
        EGL_CONTEXT_MAJOR_VERSION_KHR, 3,
        EGL_CONTEXT_MINOR_VERSION_KHR, 3,
        EGL_NONE
	};
	EGLContext egl_ctx = eglCreateContext(egl_disp, config, EGL_NO_CONTEXT, contextAttributeList);
	assert(eglMakeCurrent(egl_disp, egl_surface, egl_surface, egl_ctx) == EGL_TRUE);

	gladLoadGL();
	printf("initialised egl\n");

	/*libnx::Framebuffer fb;
	framebufferCreate(&fb, win, 684, 384, libnx::PIXEL_FORMAT_RGBA_8888, 2);
	framebufferMakeLinear(&fb);*/

	GLuint vbo, vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(screenVertices), &screenVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(offsetof(Vertex, xy)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(offsetof(Vertex, uv)));

	glBindVertexArray(0);

	GLuint vtx_shdr = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vtx_shdr, 1, &vertexShaderSource, nullptr);
	glCompileShader(vtx_shdr);
	GLuint frg_shdr = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frg_shdr, 1, &fragmentShaderSource, nullptr);
	glCompileShader(frg_shdr);

	GLuint prog = glCreateProgram();
	glAttachShader(prog, vtx_shdr);
	glAttachShader(prog, frg_shdr);
	glLinkProgram(prog);

	glDeleteShader(vtx_shdr);
	glDeleteShader(frg_shdr);

	glUseProgram(prog);
	GLint loc_proj = glGetUniformLocation(prog, "proj");
	GLint loc_tex = glGetUniformLocation(prog, "inTexture");
	GLint loc_uvoff = glGetUniformLocation(prog, "uvOffset");
	glUseProgram(0);

	glActiveTexture(GL_TEXTURE0);
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 192*2, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	char *rom_path = argv[1];

	/* the firmware settings */
    FirmwareConfig fw_config;
    NDS_GetDefaultFirmwareConfig(fw_config);
	fw_config.language = 3;
	NDS_InitFirmwareWithConfig(fw_config);

  	NDS_Init();

  	GPU->Change3DRendererByID(2);
	GPU->SetColorFormat(NDSColorFormat_BGR888_Rev);
	GPU->SetCustomFramebufferSize(GPU_FRAMEBUFFER_NATIVE_WIDTH, GPU_FRAMEBUFFER_NATIVE_HEIGHT);
  	SPU_ChangeSoundCore(SNDCORE_SWITCH, 735 * 4);

	CommonSettings.use_jit = true;
	CommonSettings.jit_max_block_size = 30;
	CommonSettings.loadToMemory = true;

	printf("loading rom %s\n", rom_path);

	if (NDS_LoadROM( rom_path, NULL, NULL) < 0) {
		printf("Error loading %s\n", rom_path);
	}

	execute = TRUE;
	u32 width, height;

	u64 frametimes[30];
	u32 frames = 0;

	printf("main loop\n");

	float x = 0.f;

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

		//profiler::frame();

		u32* src = (uint32_t*)GPU->GetDisplayInfo().masterCustomBuffer;
		//u32* dst = (u32*)libnx::framebufferBegin(&fb, &stride);

		glViewport(0, 0, 1280, 720);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_BLEND);
		glCullFace(GL_BACK);
		glDisable(GL_STENCIL_TEST);
		glFrontFace(GL_CCW);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);

		glBindVertexArray(vao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glUseProgram(prog);

		glClear(GL_COLOR_BUFFER_BIT);

		glUniform1i(loc_tex, 0);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 192*2, GL_RGBA, GL_UNSIGNED_BYTE, src);
		float proj[16];
		xm4_orthographic(proj, 0.f, 684.f, 384.f, 0.f, -1.f, 1.f);
		float translate[16];
		xm4_translatev(translate, 214.f, 0.f, 0.f);
		float res[16];
		xm4_mul(res, translate, proj);
		glUniformMatrix4fv(loc_proj, 1, GL_FALSE, &res[0]);
		glUniform1f(loc_uvoff, 0.f);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		xm4_translatev(translate, 214.f, 192.f, 0.f);
		xm4_mul(res, translate, proj);
		glUniformMatrix4fv(loc_proj, 1, GL_FALSE, &res[0]);
		glUniform1f(loc_uvoff, 0.5f);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		/*for (int y = 0; y < 192 * 2; y++) {
			for (int x = 0; x < 256; x++)
				dst[x + 214] = src[y*256+x]|(0xff<<24);
			dst += stride / 4;
		}*/

		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUseProgram(0);

		eglSwapBuffers(egl_disp, egl_surface);

		//libnx::framebufferEnd(&fb);

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
    }

    NDS_FreeROM();
    NDS_DeInit();

	deinit_switch_sound();

	libnx::socketExit();

	//libnx::framebufferClose(&fb);

	if (originalCpuFreq != 0)
	{
		libnx::pcvSetClockRate(libnx::PcvModule_Cpu, originalCpuFreq);
		libnx::pcvExit();
	}

	return 0;
}