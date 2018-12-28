#---------------------------------------------------------------------------------
.SUFFIXES:
#---------------------------------------------------------------------------------

ifeq ($(strip $(DEVKITPRO)),)
$(error "Please set DEVKITPRO in your environment. export DEVKITPRO=<path to>/devkitpro")
endif

TOPDIR ?= $(CURDIR)
include $(DEVKITPRO)/libnx/switch_rules

#---------------------------------------------------------------------------------
# TARGET is the name of the output
# BUILD is the directory where object files & intermediate files will be placed
# SOURCES is a list of directories containing source code
# DATA is a list of directories containing data files
# INCLUDES is a list of directories containing header files
# EXEFS_SRC is the optional input directory containing data copied into exefs, if anything this normally should only contain "main.npdm".
# ROMFS is the directory containing data to be added to RomFS, relative to the Makefile (Optional)
#
# NO_ICON: if set to anything, do not use icon.
# NO_NACP: if set to anything, no .nacp file is generated.
# APP_TITLE is the name of the app stored in the .nacp file (Optional)
# APP_AUTHOR is the author of the app stored in the .nacp file (Optional)
# APP_VERSION is the version of the app stored in the .nacp file (Optional)
# APP_TITLEID is the titleID of the app stored in the .nacp file (Optional)
# ICON is the filename of the icon (.jpg), relative to the project folder.
#   If not set, it attempts to use one of the following (in this order):
#     - <Project name>.jpg
#     - icon.jpg
#     - <libnx folder>/default_icon.jpg
#---------------------------------------------------------------------------------

DESMUME_SRC	:=	desmume/src

TARGET		:=	$(notdir $(CURDIR))
BUILD		:=	build
SOURCES		:=	$(DESMUME_SRC) $(DESMUME_SRC)/utils/decrypt \
				$(DESMUME_SRC)/addons $(DESMUME_SRC)/utils $(DESMUME_SRC)/utils/arm_arm64 \
				$(DESMUME_SRC)/utils/arm_arm64/emitter $(DESMUME_SRC)/utils/tinyxml \
				$(DESMUME_SRC)/utils/libfat $(DESMUME_SRC)/utils/colorspacehandler $(DESMUME_SRC)/filter \
				$(DESMUME_SRC)/metaspu $(DESMUME_SRC)/switch $(DESMUME_SRC)/libretro-common/file \
				$(DESMUME_SRC)/libretro-common/compat/ $(DESMUME_SRC)/libretro-common/features/ \
				$(DESMUME_SRC)/libretro-common/rthreads $(DESMUME_SRC)/libretro-common/encodings
DATA		:=	data
INCLUDES	:=	$(DESMUME_SRC) $(DESMUME_SRC)/libretro-common/include $(DESMUME_SRC)/frontend/windows/winpcap
EXEFS_SRC	:=	exefs_src
#ROMFS	:=	romfs

#---------------------------------------------------------------------------------
# options for code generation
#---------------------------------------------------------------------------------
ARCH	:=	-march=armv8-a -mtune=cortex-a57 -mtp=soft -fPIE

CFLAGS	:=	-g -Wall -O2 -ffunction-sections \
			$(ARCH) $(DEFINES)

CFLAGS	+=	$(INCLUDE) -D__SWITCH__ -DHAVE_LIBZ -DHAVE_JIT -DENABLE_SSE2

CXXFLAGS	:= $(CFLAGS) -fno-rtti

ASFLAGS	:=	-g $(ARCH)
LDFLAGS	=	-specs=$(DEVKITPRO)/libnx/switch.specs -g $(ARCH) -Wl,-Map,$(notdir $*.map)

LIBS	:= -lnx -lz

#---------------------------------------------------------------------------------
# list of directories containing libraries, this must be the top level containing
# include and lib
#---------------------------------------------------------------------------------
LIBDIRS	:= $(PORTLIBS) $(LIBNX)


#---------------------------------------------------------------------------------
# no real need to edit anything past this point unless you need to add additional
# rules for different file extensions
#---------------------------------------------------------------------------------
ifneq ($(BUILD),$(notdir $(CURDIR)))
#---------------------------------------------------------------------------------

export OUTPUT	:=	$(CURDIR)/$(TARGET)
export TOPDIR	:=	$(CURDIR)

export VPATH	:=	$(foreach dir,$(SOURCES),$(CURDIR)/$(dir)) \
			$(foreach dir,$(DATA),$(CURDIR)/$(dir))

export DEPSDIR	:=	$(CURDIR)/$(BUILD)

DESMUME_SOURCES := armcpu.cpp \
	arm_instructions.cpp \
	bios.cpp cp15.cpp \
	commandline.cpp \
	common.cpp \
	debug.cpp \
	Database.cpp \
	emufile.cpp encrypt.cpp FIFO.cpp \
	firmware.cpp GPU.cpp \
	mc.cpp \
	path.cpp \
	readwrite.cpp \
	wifi.cpp \
	MMU.cpp NDSSystem.cpp \
	ROMReader.cpp \
	render3D.cpp \
	rtc.cpp \
	saves.cpp \
	slot1.cpp \
	slot2.cpp \
	SPU.cpp \
	matrix.cpp \
	gfx3d.cpp \
	thumb_instructions.cpp \
	movie.cpp \
	mic.cpp \
	utils/vfat.cpp \
	utils/libfat/cache.cpp \
	utils/libfat/directory.cpp \
	utils/libfat/disc.cpp \
	utils/libfat/fatdir.cpp \
	utils/libfat/fatfile.cpp \
	utils/libfat/filetime.cpp \
	utils/libfat/file_allocation_table.cpp \
	utils/libfat/libfat.cpp \
	utils/libfat/libfat_public_api.cpp \
	utils/libfat/lock.cpp \
	utils/libfat/partition.cpp \
	utils/advanscene.cpp \
	utils/datetime.cpp \
	utils/guid.cpp \
	utils/emufat.cpp \
	utils/fsnitro.cpp \
	utils/xstring.cpp \
	utils/decrypt/crc.cpp utils/decrypt/decrypt.cpp \
	utils/decrypt/header.cpp \
	utils/task.cpp \
	utils/colorspacehandler/colorspacehandler.cpp \
	utils/dlditool.cpp \
	utils/tinyxml/tinystr.cpp \
	utils/tinyxml/tinyxml.cpp \
	utils/tinyxml/tinyxmlerror.cpp \
	utils/tinyxml/tinyxmlparser.cpp \
	addons/slot2_auto.cpp \
	addons/slot2_mpcf.cpp \
	addons/slot2_paddle.cpp \
	addons/slot2_gbagame.cpp \
	addons/slot2_none.cpp \
	addons/slot2_rumblepak.cpp \
	addons/slot2_guitarGrip.cpp \
	addons/slot2_expMemory.cpp \
	addons/slot2_piano.cpp \
	addons/slot2_passme.cpp \
	addons/slot1_none.cpp \
	addons/slot1_r4.cpp \
	addons/slot1_retail_nand.cpp \
	addons/slot1_retail_auto.cpp \
	addons/slot1_retail_mcrom.cpp \
	addons/slot1_retail_mcrom_debug.cpp \
	addons/slot1comp_mc.cpp \
	addons/slot1comp_rom.cpp \
	addons/slot1comp_protocol.cpp \
	cheatSystem.cpp \
	texcache.cpp rasterize.cpp \
	metaspu/metaspu.cpp \
	filter/2xsai.cpp \
	filter/bilinear.cpp \
	filter/deposterize.cpp \
	filter/epx.cpp \
	filter/hq2x.cpp \
	filter/hq3x.cpp \
	filter/hq4x.cpp \
	filter/lq2x.cpp \
	filter/scanline.cpp \
	filter/xbrz.cpp \
	version.cpp \
	driver.cpp \
	switch/main.cpp \
	switch/menu.cpp \
	switch/sound.cpp \
	switch/input.cpp \
	switch/profiler.cpp \
	utils/arm_arm64/arm_jit.cpp \
	utils/arm_arm64/emitter/Arm64Emitter.cpp \
	utils/arm_arm64/emitter/MathUtil.cpp

LIBRETRO_SOURCES := libretro-common/compat/compat_getopt.c \
	libretro-common/file/file_path.c \
	libretro-common/compat/compat_strl.c \
	libretro-common/features/features_cpu.c \
	libretro-common/file/retro_dirent.c \
	libretro-common/file/retro_stat.c \
	libretro-common/rthreads/async_job.c \
	libretro-common/rthreads/rsemaphore.c \
	libretro-common/rthreads/switch_thread.c \
	libretro-common/encodings/encoding_utf.c

CFILES		:= $(notdir $(LIBRETRO_SOURCES))
CPPFILES	:= $(notdir $(DESMUME_SOURCES))
SFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))
BINFILES	:=	$(foreach dir,$(DATA),$(notdir $(wildcard $(dir)/*.*)))

#---------------------------------------------------------------------------------
# use CXX for linking C++ projects, CC for standard C
#---------------------------------------------------------------------------------
ifeq ($(strip $(CPPFILES)),)
#---------------------------------------------------------------------------------
	export LD	:=	$(CC)
#---------------------------------------------------------------------------------
else
#---------------------------------------------------------------------------------
	export LD	:=	$(CXX)
#---------------------------------------------------------------------------------
endif
#---------------------------------------------------------------------------------

export OFILES_BIN	:=	$(addsuffix .o,$(BINFILES))
export OFILES_SRC	:=	$(CPPFILES:.cpp=.o) $(CFILES:.c=.o) $(SFILES:.s=.o)
export OFILES 	:=	$(OFILES_BIN) $(OFILES_SRC)
export HFILES_BIN	:=	$(addsuffix .h,$(subst .,_,$(BINFILES)))

export INCLUDE	:=	$(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
			$(foreach dir,$(LIBDIRS),-I$(dir)/include) \
			-I$(CURDIR)/$(BUILD)

export LIBPATHS	:=	$(foreach dir,$(LIBDIRS),-L$(dir)/lib)

export BUILD_EXEFS_SRC := $(TOPDIR)/$(EXEFS_SRC)

ifeq ($(strip $(ICON)),)
	icons := $(wildcard *.jpg)
	ifneq (,$(findstring $(TARGET).jpg,$(icons)))
		export APP_ICON := $(TOPDIR)/$(TARGET).jpg
	else
		ifneq (,$(findstring icon.jpg,$(icons)))
			export APP_ICON := $(TOPDIR)/icon.jpg
		endif
	endif
else
	export APP_ICON := $(TOPDIR)/$(ICON)
endif

ifeq ($(strip $(NO_ICON)),)
	export NROFLAGS += --icon=$(APP_ICON)
endif

ifeq ($(strip $(NO_NACP)),)
	export NROFLAGS += --nacp=$(CURDIR)/$(TARGET).nacp
endif

ifneq ($(APP_TITLEID),)
	export NACPFLAGS += --titleid=$(APP_TITLEID)
endif

ifneq ($(ROMFS),)
	export NROFLAGS += --romfsdir=$(CURDIR)/$(ROMFS)
endif

.PHONY: $(BUILD) clean all

#---------------------------------------------------------------------------------
all: $(BUILD)

$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

#---------------------------------------------------------------------------------
clean:
	@echo clean ...
	@rm -fr $(BUILD) $(TARGET).pfs0 $(TARGET).nso $(TARGET).nro $(TARGET).nacp $(TARGET).elf


#---------------------------------------------------------------------------------
else
.PHONY:	all

DEPENDS	:=	$(OFILES:.o=.d)

#---------------------------------------------------------------------------------
# main targets
#---------------------------------------------------------------------------------
all	:	$(OUTPUT).pfs0 $(OUTPUT).nro

$(OUTPUT).pfs0	:	$(OUTPUT).nso

$(OUTPUT).nso	:	$(OUTPUT).elf

ifeq ($(strip $(NO_NACP)),)
$(OUTPUT).nro	:	$(OUTPUT).elf $(OUTPUT).nacp
else
$(OUTPUT).nro	:	$(OUTPUT).elf
endif

$(OUTPUT).elf	:	$(OFILES)

$(OFILES_SRC)	: $(HFILES_BIN)

#---------------------------------------------------------------------------------
# you need a rule like this for each extension you use as binary data
#---------------------------------------------------------------------------------
%.bin.o	%_bin.h :	%.bin
#---------------------------------------------------------------------------------
	@echo $(notdir $<)
	@$(bin2o)

-include $(DEPENDS)

#---------------------------------------------------------------------------------------
endif
#---------------------------------------------------------------------------------------
