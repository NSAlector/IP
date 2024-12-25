#pragma once
#include <stdlib.h>
#include <string>  
#include <iostream>
#include <vector>
#include <iomanip>
#include <IProgram.hpp>
#include <ICommand.hpp>

using namespace std;
using namespace Magick;

namespace CIP
{

class HelpCommand: public ICommand
{
private:
	static ICommand *command;
	HelpCommand();
public:
	static ICommand* Initialize();
	int Execute(vector<string> args, CIP::IProgram *model);
	~HelpCommand();
};


class ExitCommand: public ICommand
{
private:
	static ICommand *command;
	ExitCommand();
public:
	static ICommand* Initialize();
	int Execute(vector<string> args, CIP::IProgram *model);
	~ExitCommand();
};

class LoadCommand: public ICommand
{
private:
	static ICommand *command;
	LoadCommand();
public:
	static ICommand* Initialize();
	int Execute(vector<string> args, CIP::IProgram *model);
	~LoadCommand();
};

class SaveCommand: public ICommand
{
private:
	static ICommand *command;
	SaveCommand();
public:
	static ICommand* Initialize();
	int Execute(vector<string> args, CIP::IProgram *model);
	~SaveCommand();
};


class BlurCommand: public ICommand
{
private:
	static ICommand *command;
	BlurCommand();
public:
	static ICommand* Initialize();
	int Execute(vector<string> args, CIP::IProgram *model);
	~BlurCommand();
};

class ResizeCommand: public ICommand
{
private:
	static ICommand *command;
	ResizeCommand();
public:
	static ICommand* Initialize();
	int Execute(vector<string> args, CIP::IProgram *model);
	~ResizeCommand();
};

};
