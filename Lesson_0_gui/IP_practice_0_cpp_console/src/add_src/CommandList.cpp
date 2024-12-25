#include <CommandList.hpp>

CIP::ICommandList* CIP::ConsoleCommandList::instance = 0;

CIP::ConsoleCommandList::ConsoleCommandList()
{
	try{
	//insert EXIT
	comm_names.push_back(vector<string>{"q", "quit", "exit"});
	comm_info.push_back(vector<string>{"EXIT FROM PROGRAM", 
						"ANOTHER PARAMETERS ARE IGNORED"});
	for(auto name: comm_names.back())
		commands.insert(pair<string, CIP::ICommand*>(name, CIP::ExitCommand::Initialize()));

	//insert HELP
	comm_names.push_back(vector<string>{"h", "help"});
	comm_info.push_back(vector<string>{"HELP", 
					"ANOTHER PARAMETERS ARE IGNORED"});
	for(auto name: comm_names.back())
		commands.insert(pair<string, CIP::ICommand*>(name,CIP::HelpCommand::Initialize()));

	
	//insert LOAD
	comm_names.push_back(vector<string>{"ld", "load"});
	comm_info.push_back(vector<string>{"IMAGE LOADING",
				"PARAMETERS:",
				"<name> - image name",
				"<filename> - loading image file path" });
	for(auto name: comm_names.back())
		commands.insert(pair<string, CIP::ICommand*>(name,CIP::LoadCommand::Initialize()));

	//insert SAVE
	comm_names.push_back(vector<string>{"s", "store"});
	comm_info.push_back(vector<string>{"IMAGE SAVING",
				"PARAMETERS:",
				"<name> - image name",
				"<filename> - saving file path for image"});
	for(auto name: comm_names.back())
		commands.insert(pair<string, CIP::ICommand*>(name,CIP::SaveCommand::Initialize()));

	//insert RESIZE
	comm_names.push_back(vector<string>{"resize"});
	comm_info.push_back(vector<string>{"IMAGE RESIZING",
				"PARAMETERS:",
				"<from_name> - source image name",
				"<to_name> - destination image name",
				"<new_width> ", "<new_height> " });
	for(auto name: comm_names.back())
		commands.insert(pair<string, CIP::ICommand*>(name,CIP::ResizeCommand::Initialize()));
	
	//insert BLUR
	comm_names.push_back(vector<string>{"blur"});
	comm_info.push_back(vector<string>{ "IMAGE BLURING",
				"PARAMETERS:",
				"<from_name> - source image name",
				"<to_name> - destination image name",
				"<size> - bluring area size" });
	for(auto name: comm_names.back())
		commands.insert(pair<string, CIP::ICommand*>(name,CIP::BlurCommand::Initialize()));
	}
	catch(...)
	{
		throw ;
	}
}

int CIP::ConsoleCommandList::Help()
{
	try{
	int widths[2] = {5,0};
	cout<<setw(widths[0])<<" ";
	cout<<"###---HELP INFORMATION---###"<<endl;
	
	for(int i=0; i < comm_names.size(); ++i)
	{
		cout<<setw(widths[0])<<" ";
		for(auto name: comm_names[i])
		{
			cout<< name<<" "; 
		}

		vector<string>::iterator it_info = comm_info[i].begin();

		for(;it_info != comm_info[i].end(); ++it_info)
		{
			
			cout<<std::left;
			cout<<setw(widths[0])<<" ";
			if(it_info != comm_info[i].begin()) cout<<"\t\t";
			cout<<"\t";
			cout<< *it_info<<endl;
		}
		cout<<endl;
	}
	}
	catch(...)
	{
		throw ;
		return 1;
	}
	return 0;
}

CIP::ICommandList* CIP::ConsoleCommandList::InitCommands()
{
	if(instance== NULL) 
	{
		instance = new CIP::ConsoleCommandList();

	}
	return instance;
}

int CIP::ConsoleCommandList::Execute(vector<string> args, CIP::IProgram *model) 
{
	try{
	std::map<string,CIP::ICommand*>::iterator it;
	it = commands.find(args[0]);
	if (it != commands.end())
		return it->second->Execute(args, model);
	else return 1;
	}
	catch(...)
	{
		throw ;
		return 1;
	}
	return 0;
}
CIP::ConsoleCommandList::~ConsoleCommandList()
{	
}
