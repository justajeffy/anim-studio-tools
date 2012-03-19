/*
 * hello_world definition
 */

#ifndef __HELLO_WORLD_H__
#define __HELLO_WORLD_H__

#include <string>


class Greeter
{

public:
	Greeter(std::string greeting="Hello");
	~Greeter();

	void greet(std::string who="World");


protected:

	std::string greeting;

};


#endif /*__HELLO_WORLD_H__*/

