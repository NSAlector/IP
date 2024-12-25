#pragma once
#include <string>  
#include <Magick++.h>

using namespace std;
using namespace Magick;

namespace CIP
{

class IProgram
{
public:
	virtual int Help()=0;	
	virtual int SetVariable(string name, Image img) = 0;
	virtual Image GetVariable(string name) = 0;
	virtual int Run(int argc, char **argv) = 0;
};
};
