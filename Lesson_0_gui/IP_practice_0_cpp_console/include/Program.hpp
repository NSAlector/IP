#pragma once
#include <vector>
#include <string>  
#include <sstream>
#include <iostream>
#include <map>
#include <IProgram.hpp>
//#include <Command.hpp>
#include <CommandList.hpp>

using namespace std;
using namespace Magick;

namespace CIP
{

class Program:public IProgram
{
private:
	map<string,Image> variables;
	CIP::ICommandList* com_list;
public:
	Program();
	int Help();
	int SetVariable(string name, Image img);
	Image GetVariable(string name);
	int Run(int argc, char **argv);
	~Program();
};
};
