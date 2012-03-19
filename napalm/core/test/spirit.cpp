//#define BOOST_SPIRIT_DEBUG

//#include "napalm/core/parsing/parseXml.h"
//#include "napalm/core/Table.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char** argv)
{
	/*
	if ( argc != 2 )
	{
		std::cerr << "No input" << std::endl;
		return -1;
	}
	std::ifstream ifs( argv[1] );
	std::string input, error;
	input.assign( std::istreambuf_iterator<char>( ifs ), std::istreambuf_iterator<char>() );

	napalm::object_table_ptr t = napalm::parseXml( input, &error );
	std::cout << error << std::endl;

	napalm::object_rawptr_set printed;
	t->dump(std::cout, printed);
	*/
	return 0;
}
