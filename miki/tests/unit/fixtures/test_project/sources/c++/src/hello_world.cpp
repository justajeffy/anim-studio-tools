/*
 * hello_world implementation
 */

#include "hello_world.h"
#include <iostream>
#include <string>


/* Constructor */
Greeter::Greeter(std::string greeting)
{
    this->greeting = greeting;
}

/* Destructor */
Greeter::~Greeter()
{
}


/* Greet an entity */
void Greeter::greet(std::string who)
{
	std::cout << greeting << who << std::endl;
}
