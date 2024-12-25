#include <Program.hpp>

CIP::Program::Program()
{
	com_list = CIP::ConsoleCommandList::InitCommands();
		
}

int CIP::Program::Help()
{
	com_list->Help();	
	return 0;		
}

int CIP::Program::SetVariable(string name, Image img)
{
	try{
	std::map<string,Image>::iterator it;
	it = variables.find(name);

	if(it !=variables.end())
		it->second = img;
	else
		variables.insert(pair<string, Image>(name, img));
	}
	catch(...)
	{
		throw ;
		return 1;
	}
	return 0;
}

Image CIP::Program::GetVariable(string name)
{
	try{
	std::map<string,Image>::iterator it;
	it = variables.find(name);

	if(it !=variables.end()) return it->second;
	else return NULL;
	}
	catch(...)
	{
		throw ;
	}
}

int CIP::Program::Run(int argc, char **argv)
{
	InitializeMagick(NULL);

	for(string line; getline(cin, line); )
	{
	istringstream iss(line);
	vector<string> args;
       	for (std::string arg; iss >> arg; )
		args.push_back(arg);
	//cout<<com_list->Execute(args,this)<<endl;
	if(com_list->Execute(args,this) == 0) cout<<"DONE"<<endl;
	else cout<<"ERROR COMMAND"<<endl;

	}

	return 0;
}
CIP::Program::~Program()
{
}
