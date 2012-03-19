#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:get_params" );

#include "get_params.h"


//-------------------------------------------------------------------------------------------------
template< typename BUFFER, typename FIXED_RANGE >
void getBufferValues(	const napalm::ObjectTable& o,
						std::vector< std::string >& tbuffers,
						std::vector< FIXED_RANGE >& vbuffers,
						std::vector< RtToken >& tokens,
						std::vector< RtPointer >& parms,
						std::vector< int >& strides )
{
	napalm::ObjectTable::const_iterator i( o.begin() ), e( o.end() );
	for ( ; i != e ; ++i )
	{
		// try to get as a buffer of the appropriate type
		BUFFER b;

		if ( i.getValue( b ) )
		{
			napalm::c_attrib_table_ptr attrs = b->getAttribs();
			std::string token;
			if ( attrs->getEntry( "token", token ) )
			{
				DRD_LOG_DEBUG( L, "found buffer: " << token );

				tbuffers.push_back( token );
				tokens.push_back( (RtToken) tbuffers.back().c_str() );
				vbuffers.push_back( b->r() );
				parms.push_back( ( RtPointer ) & vbuffers.back()[ 0 ] );
				strides.push_back( sizeof( typename FIXED_RANGE::value_type ) / sizeof( float ) );
			}

			continue;
		}
	}
}


//-------------------------------------------------------------------------------------------------
template< typename T >
void getAttribValues( const napalm::ObjectTable& o,
						std::vector< std::string >& tbuffers,
						std::vector< T >& vbuffers,
						std::vector< RtToken >& tokens,
						std::vector< RtPointer >& parms,
						std::vector< int >& strides )
{
	napalm::ObjectTable::const_iterator i( o.begin() ), e( o.end() );
	for ( ; i != e ; ++i )
	{
		napalm::c_attrib_table_ptr a;
		if( i.getValue( a ) )
		{
			std::string token;
			if ( a->getEntry( "token", token ) )
			{
				T value;
				if( a->getEntry( "value", value ) )
				{
					DRD_LOG_DEBUG( L, "found attrib: " << token << ": " << value );

					tbuffers.push_back( token );
					tokens.push_back( (RtToken) tbuffers.back().c_str() );
					vbuffers.push_back( value );
					parms.push_back( ( RtPointer ) & vbuffers.back() );
					strides.push_back( 0 );
				}
			}
		}
	}
}


//-------------------------------------------------------------------------------------------------
// specialization for strings
template<>
void getAttribValues( const napalm::ObjectTable& o,
						std::vector< std::string >& tbuffers,
						std::vector< std::vector< const char* > >& vbuffers,
						std::vector< RtToken >& tokens,
						std::vector< RtPointer >& parms,
						std::vector< int >& strides )
{
	napalm::ObjectTable::const_iterator i( o.begin() ), e( o.end() );
	for ( ; i != e ; ++i )
	{
		napalm::c_attrib_table_ptr a;
		if( i.getValue( a ) )
		{
			std::string token;
			if ( a->getEntry( "token", token ) )
			{
				std::string value;
				if( a->getEntry( "value", value ) )
				{
					DRD_LOG_DEBUG( L, "found attrib: " << token << ": " << value );

					tbuffers.push_back( token );
					tokens.push_back( (RtToken) tbuffers.back().c_str() );

					// we'll store the string value in tbuffers for convenience
					tbuffers.push_back( value );

					// now to maintain a list of const char*'s
					std::vector< const char* > data;
					data.push_back( tbuffers.back().c_str() );
					vbuffers.push_back( data );
					parms.push_back( ( RtPointer ) & vbuffers.back()[0] );
					strides.push_back( 0 );
				}
			}
		}
	}
}


//-------------------------------------------------------------------------------------------------
void napalm_delight::getParams( const napalm::ObjectTable& o, ParamStruct& dst )
{
	DRD_LOG_DEBUG( L, __FUNCTION__ << ": " << o );

	getAttribValues( o, dst.tbuffers, dst.fvals, dst.tokens, dst.parms, dst.strides );
	getAttribValues( o, dst.tbuffers, dst.vvals, dst.tokens, dst.parms, dst.strides );
	getAttribValues( o, dst.tbuffers, dst.stringVals, dst.tokens, dst.parms, dst.strides );

	getBufferValues< napalm::FloatBufferCPtr, napalm::FloatBuffer::r_type >( o, dst.tbuffers, dst.fbuffers, dst.tokens, dst.parms, dst.strides );
	getBufferValues< napalm::V3fBufferCPtr, napalm::V3fBuffer::r_type >( o, dst.tbuffers, dst.vbuffers, dst.tokens, dst.parms, dst.strides );
}

