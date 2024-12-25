#pragma once
#include <ICommandList.hpp>

using namespace std;
using namespace Magick;

namespace CIP
{

class ConsoleCommandList:public ICommandList
{
private:
	vector<vector<string>> comm_names;
	vector<vector<string>> comm_info;
	map<string, CIP::ICommand * > commands;

	static CIP::ICommandList* instance;
	ConsoleCommandList();

public:
	int Help();
	static CIP::ICommandList* InitCommands();
	virtual int Execute(vector<string> args, CIP::IProgram *model);
	~ConsoleCommandList();
};

};
