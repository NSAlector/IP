#pragma once
#include <Magick++.h>
#include <Command.hpp>
#include <IProgram.hpp>

using namespace std;
using namespace Magick;

namespace CIP
{

class ICommandList
{

public:
	virtual int Help()=0;	
	virtual int Execute(vector<string> args, CIP::IProgram *model) = 0;
};

};
