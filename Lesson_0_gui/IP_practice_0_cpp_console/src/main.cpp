#include "Program.hpp"

int main(int argc, char **argv)
{
	CIP::IProgram *prog = new CIP::Program();
	prog->Run(argc, argv);

	return 0;
}
