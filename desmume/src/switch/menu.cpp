#include <dirent.h>

#include <vector>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include<switch.h>

//Very rough implementation of a rom selector... Yeah I know its ugly
char* menu_FileBrowser() 
{
	std::vector<std::string> entries;
	DIR *directory;
	dirent *entry;
	directory = opendir("sdmc:/switch/desmume/roms");

	while(entry = readdir(directory))
	{
		if(entry->d_type == DT_DIR) continue;

		char *extension = strrchr(entry->d_name,'.');

		if(extension != NULL)
		{
			if(strcmp(extension, ".nds") == 0)
				entries.push_back(entry->d_name);
		}
	}

	if(entries.empty())
		return NULL;

	int count, cursor = 0;
	uint32_t keysDown;

	while(true) 
	{
		hidScanInput();
		keysDown = hidKeysDown(CONTROLLER_P1_AUTO);

		if(keysDown & KEY_A)
			break;

		if(keysDown & KEY_DOWN)
			cursor++;

		if(keysDown & KEY_UP)
			cursor--;

		if(cursor < 0)
			cursor = 0;

		if(cursor > entries.size() - 1)
			cursor = entries.size() - 1;

		count = 0;

		printf(CONSOLE_ESC(0;0H));

		for( auto it = entries.begin(); it != entries.end(); it++)
		{
			if(cursor == count)
				printf(CONSOLE_ESC(4m));
			else
				printf(CONSOLE_ESC(0m));

			printf("%s\n", it->c_str());

			count++;
		}

		printf(CONSOLE_ESC(0m));

		//printf(CONSOLE_ESC(40;0H)"Press (X) to %s Sound\n", UserConfiguration.soundEnabled ? "disable" : " enable");
		printf("Press Left or Right DPAD to change frameSkip\n");
		//printf("Current frameSkip value: %u\n", UserConfiguration.frameSkip);

	}

	printf(CONSOLE_ESC(0;0H));

	char *filename = (char*)malloc(4096);
	sprintf(filename, "sdmc:/switch/desmume/roms/%s", entries[cursor].c_str());

	return filename;
}

//Uninplemented
int menu_Init()
{
	return 0;
}

//Uninplemented
//This is where all the tabs(rom selection, save states, settings, etc) will be handled
int menu_Display()
{
	return 0;
}