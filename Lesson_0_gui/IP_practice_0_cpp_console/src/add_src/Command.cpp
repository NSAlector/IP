#include <Command.hpp>


CIP::ICommand* CIP::ExitCommand::command = 0;
CIP::ICommand* CIP::HelpCommand::command = 0;
CIP::ICommand* CIP::LoadCommand::command = 0;
CIP::ICommand* CIP::SaveCommand::command = 0;
CIP::ICommand* CIP::BlurCommand::command = 0;
CIP::ICommand* CIP::ResizeCommand::command = 0;

int CIP::ExitCommand::Execute(vector<string> args, IProgram *model)
{
	exit(0);
	return 0;
}


int CIP::HelpCommand::Execute(vector<string> args, IProgram *model)
{
	model->Help();
	return 0;
}

int CIP::LoadCommand::Execute(vector<string> args, IProgram *model)
{
	try{
	if(args.size()>=2)
	{ 
	Image img;
	img.read(args[2]);
	model->SetVariable(args[1], img);
	return 0;
	}else return 1;
	}
	catch(...)
	{
		throw ;
		return 1;
	}
}

int CIP::SaveCommand::Execute(vector<string> args, IProgram *model)
{
	try
	{
	if(args.size() >=2){
	Image img = model->GetVariable(args[1]);
	img.write(args[2]);
	return 0;
	}else return 1;
	}
	catch(...)
	{
		throw ;
		return 1;
	}
}

int CIP::BlurCommand::Execute(vector<string> args, IProgram *model)
{
	try
	{
	if(args.size() >=4) {
	Image img(model->GetVariable(args[1]));
	img.blur(stof(args[3]));
	model->SetVariable(args[2], img);
	return 0;
	}else return 1;
	}
	catch(...)
	{
		throw ;
		return 1;
	}
}

int CIP::ResizeCommand::Execute(vector<string> args, IProgram *model)
{
	try{
	if(args.size() >=5) {

	Image img(model->GetVariable(args[1]));
	img.scale(Magick::Geometry(stoi(args[3]),stoi(args[4])));
	model->SetVariable(args[2], img);
	return 0;
	} else return 1;
	}
	catch(...)
	{
		throw ;
		return 1;
	}
}


CIP::HelpCommand::HelpCommand()
{
}

CIP::ICommand* CIP::HelpCommand::Initialize()
{
	if(command == 0)
		command = new HelpCommand();
	return command;
}

CIP::HelpCommand::~HelpCommand()
{
}

CIP::ExitCommand::ExitCommand()
{
}

CIP::ICommand* CIP::ExitCommand::Initialize()
{
	if(command == 0)
		command = new ExitCommand();
	return command;
}

CIP::ExitCommand::~ExitCommand()
{
}

CIP::LoadCommand::LoadCommand()
{
}

CIP::ICommand* CIP::LoadCommand::Initialize()
{
	if(command == 0)
		command = new LoadCommand();
	return command;
}

CIP::LoadCommand::~LoadCommand()
{
}

CIP::SaveCommand::SaveCommand()
{
}

CIP::ICommand* CIP::SaveCommand::Initialize()
{
	if(command == 0)
		command = new SaveCommand();
	return command;
}

CIP::SaveCommand::~SaveCommand()
{
}

CIP::BlurCommand::BlurCommand()
{
}

CIP::ICommand* CIP::BlurCommand::Initialize()
{
	if(command == 0)
		command = new BlurCommand();
	return command;
}

CIP::BlurCommand::~BlurCommand()
{
}

CIP::ResizeCommand::ResizeCommand()
{
}

CIP::ICommand* CIP::ResizeCommand::Initialize()
{
	if(command == 0)
		command = new ResizeCommand();
	return command;
}

CIP::ResizeCommand::~ResizeCommand()
{
}

