#include "log.h"

#include <napalmImageIO/image.h>

#include <napalm/core/Attribute.h>
#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>
#include <napalm/core/exceptions.h>

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace napalm;
using namespace napalm_image_io;

bool napalm_image_io::isValid( 	c_object_table_ptr t,
								bool report )
{
	int xres, yres;
	if ( !t->getEntry( "xres", xres ) )
	{
		if ( report ) std::cerr << "# doesn't contain xres" << std::endl;
		return false;
	}
	if ( !t->getEntry( "yres", yres ) )
	{
		if ( report ) std::cerr << "# doesn't contain yres" << std::endl;
		return false;
	}

	{
		FloatBufferCPtr pixels;
		if ( t->getEntry( "pixels", pixels ) )
		{
			if ( pixels->size() == xres * yres ) return true;
			else
			{
				if ( report ) std::cerr << "# pixel data has incorrect size" << std::endl;
				return false;
			}
		}
	}

	{
		V3fBufferCPtr pixels;
		if ( t->getEntry( "pixels", pixels ) )
		{
			if ( pixels->size() == xres * yres ) return true;
			else
			{
				if ( report ) std::cerr << "# pixel data has incorrect size" << std::endl;
				return false;
			}
		}
	}

	{
		V4fBufferCPtr pixels;
		if ( t->getEntry( "pixels", pixels ) )
		{
			if ( pixels->size() == xres * yres ) return true;
			else
			{
				if ( report ) std::cerr << "# pixel data has incorrect size" << std::endl;
				return false;
			}
		}
	}

	if ( report ) std::cerr << "# can't get float/V3f/V4f pixel data " << std::endl;

}

template< typename T >
void writeImage( 	int xres,
					int yres,
					int channels,
					TypeDesc dest_format,
					const std::string& filePath,
					const T& pixels )
{
	ImageSpec spec( xres, yres, channels, dest_format );

	ImageOutput *out = ImageOutput::create( filePath );
	if ( !out ) return;

	out->open( filePath, spec );
	out->write_image( TypeDesc::FLOAT, &pixels->r()[ 0 ] );
	out->close();
	delete out;
}

void napalm_image_io::write( 	c_object_table_ptr t,
								const std::string& filePath,
								const std::string& format )
{
	if ( !isValid( t, false ) ) throw NapalmError( "table is not valid" );

	int xres, yres;
	t->getEntry( "xres", xres );
	t->getEntry( "yres", yres );

	// string -> TypeDesc
	TypeDesc dest_format( format.c_str() );

	// try to write a mono image
	{
		FloatBufferCPtr pixels;
		if ( t->getEntry( "pixels", pixels ) )
		{
			writeImage( xres, yres, 1, dest_format, filePath, pixels );
			return;
		}
	}

	// try to write an RGB image
	{
		V3fBufferCPtr pixels;
		if ( t->getEntry( "pixels", pixels ) )
		{
			writeImage( xres, yres, 3, dest_format, filePath, pixels );
			return;
		}
	}

	// try to write an RGBA image
	{
		V4fBufferCPtr pixels;
		if ( t->getEntry( "pixels", pixels ) )
		{
			writeImage( xres, yres, 4, dest_format, filePath, pixels );
			return;
		}
	}

	throw NapalmError( "unsupported pixel format" );
}

object_table_ptr napalm_image_io::read( const std::string& filePath )
{
	object_table_ptr t( new ObjectTable() );
	ImageInput *in = ImageInput::create( filePath );
	if ( !in ) throw NapalmError( "unable to load file" );

	int xres, yres, channels;
	ImageSpec spec;
	in->open( filePath, spec );
	xres = spec.width;
	yres = spec.height;
	channels = spec.nchannels;

	t->setEntry( "xres", xres );
	t->setEntry( "yres", yres );

	if ( channels == 1 )
	{
		// read a mono image
		FloatBufferPtr buf( new FloatBuffer( xres * yres ) );
		in->read_image( TypeDesc::FLOAT, &buf->w()[ 0 ] );
		t->setEntry( "pixels", buf );
	}
	else if ( channels == 3 )
	{
		// read an RGB image
		V3fBufferPtr buf( new V3fBuffer( xres * yres ) );
		in->read_image( TypeDesc::FLOAT, &buf->w()[ 0 ] );
		t->setEntry( "pixels", buf );
	}
	else if ( channels == 4 )
	{
		// read an RGBA image
		V4fBufferPtr buf( new V4fBuffer( xres * yres ) );
		in->read_image( TypeDesc::FLOAT, &buf->w()[ 0 ] );
		t->setEntry( "pixels", buf );
	}
	else
	{
		throw NapalmError( "unsupported number of channels" );
	}

	in->close();
	delete in;
	return t;
}

