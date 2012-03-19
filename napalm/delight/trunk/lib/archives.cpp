#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:archives" );

#include "get_params.h"

#include "napalmDelight/render.h"
#include "napalmDelight/exceptions.h"
#include "napalmDelight/type_conversion.h"

#include <napalm/core/Attribute.h>

#include <boost/foreach.hpp>

using namespace napalm;
using namespace napalm_delight;


void napalm_delight::archives( const ObjectTable& archivesBuffers )
{
	DRD_LOG_DEBUG( L, "archive time samples: " << archivesBuffers.size() );

	if( archivesBuffers.size() == 0 )
	{
		DRD_LOG_ERROR( L, "can't find data in archives call" );
		return;
	}

	int emitInWorldSpace = 0;

	std::vector< int > archiveCounts;
	std::vector< float > timeSamples;

	// these helpers allow convenient access to napalm types
	std::vector< M44fBufferCPtr > xformsHelper;

	for( int i = 0; i < archivesBuffers.size(); ++i )
	{
		object_table_ptr t;
		archivesBuffers.getEntry( i, t );

		// query the emitInWorldSpace if it's there
		t->getEntry( "emitInWorldSpace", emitInWorldSpace );

		{
			int firstN;
			if( !t->getEntry( "firstN", firstN ) ){
				DRD_LOG_ERROR( L, "error extracting firstN from archives table" );
				return;
			}
			archiveCounts.push_back( firstN );
		}

		{
			float time;
			if( !t->getEntry("time",time) ){
				DRD_LOG_ERROR( L, "error extracting time sample from archives table" );
				return;
			}
			timeSamples.push_back( time );
		}

		{
			M44fBufferCPtr xforms;
			if( !t->getEntry( "transform", xforms ) )
			{
				DRD_LOG_ERROR( L, "error extracting transform from archives table" );
				return;
			}
			xformsHelper.push_back( xforms );
		}
	}

	DRD_LOG_DEBUG( L, "rendering " << archiveCounts[0] << " archives" );

	// set up non-motion blurred helpers
	StringBuffer::r_type recordIter;
	V3fBuffer::r_type OsIter;
	ParamStruct p;
	{
		object_table_ptr t;
		archivesBuffers.getEntry( 0, t );
		getParams( *t, p );

		{
			StringBufferCPtr temp;
			t->getEntry( "record", temp );
			recordIter = temp->r();
		}
		{
			V3fBufferCPtr temp;
			t->getEntry( "Os", temp );
			OsIter = temp->r();
		}
	}

	const int timeSampleCount = timeSamples.size();
	const bool doBlur = timeSamples.size() > 0;
	const int tokenCount = p.tokens.size();

	// for each archive
	for( int a = 0; a < archiveCounts[0]; ++a )
	{
		RiAttributeBegin();
		{
			// emit attributes
			assert( p.tokens.size() == p.strides.size() );
			for( int i = 0; i < tokenCount; ++i )
			{
				// skip Os as it will be handled individually
				if( strcmp( p.tokens[ i ], "Os" ) == 0 ) continue;

				RiAttribute( "user", p.tokens[ i ], RtPointer((float*)p.parms[ i ] + a * p.strides[ i ]), RI_NULL );
				//DRD_LOG_VERBATIM( L, "token: " << p.tokens[ i ] << " stride: " << p.strides[i] );
			}

			RiTransformBegin();
			{
				RiOpacity( (RtFloat*)&OsIter[a].x );

				// blurred transform
				if( doBlur ) RiMotionBeginV( timeSamples.size(), &timeSamples[0] );

				for( int ts = 0; ts < timeSampleCount; ++ts )
				{
					// extract and convert the transform
					const Imath::M44f& xform = xformsHelper[ts]->r()[a];
					// DRD_LOG_VERBATIM( L, "xform[" << ts << "] (time: " << timeSamples[ts] << "):\n" << xform );
					RtMatrix rixform;
					convert( xform, rixform );

					// now emit
					if( emitInWorldSpace )
					{
						//DRD_LOG_VERBATIM( L, "emitting in world space" );
						RiTransform( rixform );
					} else {
						RiConcatTransform( rixform );
					}
				}

				if( doBlur ) RiMotionEnd();

				// extract the emission call
				{
					const std::string& record = recordIter[ a ];
					RiArchiveRecord( "verbatim", record.c_str() );
				}

			}

			RiTransformEnd();
		}
		RiAttributeEnd();
	}
}

