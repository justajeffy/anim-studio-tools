

#include "napalmDelight/SampleSx.h"

#include <napalm/parsing/parsePythonDict.h>

#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>

#include <iostream>

using namespace napalm;
using namespace napalm_delight;


object_table_ptr constructSrc( int n )
{
	object_table_ptr result( new ObjectTable() );

	V3fBufferPtr P( new V3fBuffer( n ) );
	result->setEntry( "P", P );

	FloatBufferPtr s( new FloatBuffer( n ) );
	result->setEntry( "s", s );

	FloatBufferPtr t( new FloatBuffer( n ) );
	result->setEntry( "t", t );

	return result;
}


void dumpVec( const V3fBuffer& src )
{
	V3fBuffer::r_type r = src.r();
	for( int i = 0; i < r.size(); ++i )
	{
		std::cout << r[i] << " ";
	}
	std::cout << std::endl;
}


int main()
{
	// set up the napalm table for shader args
	const char* paramstr = "{'name':'my_tex_lookup','params':{'col':[0.8,1.0,0.8],'my_strings':['string','array','argument'],'texturename':['/drd/depts/rnd/test_data/drd.tdl']}}";
	object_table_ptr params = parsePythonDict( paramstr );

	// construct our sx sampler
	SampleSx sx( params, 0 );

	// populate the input variables
	object_table_ptr src = constructSrc( 10 );

	// this is the result we want
	ObjectTable aovs;
	aovs.setEntry(0, "Ci" );

	// empty result table
	object_table_ptr dst( new ObjectTable() );

	// compute
	sx.sample( *src, aovs, *dst );

	// see what we got for the result
	dst->Object::dump();
	V3fBufferPtr Ci;
	dst->getEntry( "Ci", Ci );
	dumpVec( *Ci );

	std::cout << "done" << std::endl;
}
