#pragma once
#include <string>  
#include <vector>
#include <IProgram.hpp>

using namespace std;

namespace CIP
{

class ICommand
{
public:
	virtual int Execute(vector<string> args, CIP::IProgram *model) = 0;
};

};
