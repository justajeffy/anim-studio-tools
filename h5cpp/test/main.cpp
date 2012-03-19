#include <h5cpp/h5cpp.h>
#include <iostream>
#include <stdlib.h>

using namespace h5cpp;


int main(int argc, char** argv)
{
	const char* fname = "/tmp/test.h5";
	h5cpp::system::init();

	std::cout << "creating h5 file..." << std::endl;
	{
		shared_hid_file file = file::create(fname);
		attrib::create(file, "fixed_len_str", fl_string("2.1.0"));
		attrib::create(file, "FPS", long(24));
		attrib::create(file, "a_float", 1.0);
		attrib::create(file, "another_float", 11.0);

		std::cout << "creating group..." << std::endl;
		{
			shared_hid_group gp = group::create(file, "group1");
			attrib::create(gp, "variable_len_str", vl_string("feefifofum"));
			attrib::create(gp, "number", 566);
		}
	}

	std::cout << "dumping result..." << std::endl;
	std::string cmd("h5dump ");
	cmd += fname;
	::system(cmd.c_str());

	return 0;
}
