#ifndef LIST_H
#define LIST_H

//#include <boost/tuple/tuple_comparison.hpp>
//#include <boost/utility.hpp>
#include <cstdarg>
#include <vector>
#include "Object.h"
#include "Attribute.h"
#include "TypedAttribute.h"
#include "impl/ValueWrapper.hpp"
#include "exceptions.h"
#include "meta.hpp"


namespace napalm {

	/*
	 * @class AttributeList
	 * @brief
	 */
	class AttributeList
	{
	public:
		typedef Attribute										value_class_type;
		typedef boost::shared_ptr<Attribute>					value_ptr;

		typedef std::vector<value_ptr>							container;
		typedef container::iterator								iterator;
		typedef container::const_iterator						const_iterator;
		typedef Wrapper<Attribute>								wrap;

		AttributeList(){}

		AttributeList( const AttributeList & a_AttributeList )
		{
			for( const_iterator it = a_AttributeList.m_contents.begin(); it != a_AttributeList.m_contents.end(); ++it )
				push_back( *it );
		}

		template<typename T>
		explicit AttributeList( const T & a_T,
			typename boost::enable_if<is_napalm_attrib_type<T>, void *>::type v = 0 )
		{
			push_back( value_ptr( new TypedAttribute<T>( a_T )));
		}


		AttributeList( const wrap & p0, const wrap & p1,
			  const wrap & p2 = wrap(), const wrap & p3 = wrap(), const wrap & p4 = wrap(),
			  const wrap & p5 = wrap(), const wrap & p6 = wrap(), const wrap & p7 = wrap() )
		{
			if( p0.isNull() ) { return; } else { m_contents.push_back( p0.get() ); }
			if( p1.isNull() ) { return; } else { m_contents.push_back( p1.get() ); }
			if( p2.isNull() ) { return; } else { m_contents.push_back( p2.get() ); }
			if( p3.isNull() ) { return; } else { m_contents.push_back( p3.get() ); }
			if( p4.isNull() ) { return; } else { m_contents.push_back( p4.get() ); }
			if( p5.isNull() ) { return; } else { m_contents.push_back( p5.get() ); }
			if( p6.isNull() ) { return; } else { m_contents.push_back( p6.get() ); }
			if( p7.isNull() ) { return; } else { m_contents.push_back( p7.get() ); }
		}

		iterator begin()
		{
			return m_contents.begin();
		}

		const_iterator begin() const
		{
			return m_contents.begin();
		}

		iterator end()
		{
			return m_contents.end();
		}

		const_iterator end() const
		{
			return m_contents.end();
		}

		value_ptr& operator[](int i)
		{
			unsigned int nelems = m_contents.size();
			if(nelems == 0)
			{
				throw "Trying to get element of empty array...";
			}
			else
			{
				return (i>=0)?
					m_contents[i%nelems] :
					m_contents[((-i/nelems)*nelems + nelems) + i];
			}
		}

		const value_ptr& operator[](int i) const
		{
			unsigned int nelems = m_contents.size();
			if(nelems == 0)
			{
				throw "Trying to get element of empty array...";
			}
			else
			{
				return (i>=0)?
					m_contents[i%nelems] :
					m_contents[((-i/nelems)*nelems + nelems) + i];
			}
		}

		iterator erase( iterator i )
		{
			return m_contents.erase( i );
		}

		void insert( iterator i, value_ptr v )
		{
			m_contents.insert( i, v );
		}

		template< typename T>
		typename boost::enable_if<is_napalm_attrib_type<T>, void>::type
		push_back( const T & a_ele )
		{
			m_contents.push_back( value_ptr( new TypedAttribute<T>(a_ele )));
		}

		void push_back( const char * a_ele )
		{
			m_contents.push_back( value_ptr( new TypedAttribute<std::string>(a_ele )) );
		}

		void push_back( const value_ptr& a_ele )
		{
			m_contents.push_back( a_ele );
		}

		bool empty() const
		{
			return m_contents.empty();
		}

		void clear()
		{
			m_contents.clear();
		}

		unsigned int size() const
		{
			return m_contents.size();
		}

		template<typename T>
		bool extractEntry( unsigned int index, T& value );

		template<typename T>
		T extractEntry( unsigned int index );

		template<typename T>
		T extractEntryWithDefault( unsigned int index, const T & defaultValue );

		AttributeList* clone(object_clone_map& cloned) const;

		std::ostream& str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type = util::DEFAULT) const;

		std::ostream& dump(std::ostream& os, object_rawptr_set& printed) const;

		bool operator==(const AttributeList& a_Other) const;
		bool operator<(const AttributeList & a_Other) const;

	protected:

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive& ar, const unsigned int version);

	protected:

		container m_contents;
	};


// Definition of
// bool AttributeList::extractEntry( unsigned int index, T& value )
// is in Dispatcher.h. This is because it uses the dispatcher & the dispatcher references
// AttributeList.

template<typename T>
T AttributeList::extractEntry( unsigned int index )
{
	T rv;
	if( !extractEntry(index, rv) )
	{
		throw NapalmError("Can't extract entry");
	}
	return rv;
}


template<typename T>
T AttributeList::extractEntryWithDefault( unsigned int index, const T & defaultValue )
{
	T rv;
	if( !extractEntry(index, rv) )
	{
		return defaultValue;
	}
	return rv;
}

template<class Archive>
void AttributeList::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("contents", m_contents);
}

namespace util {
	template<>
	struct to_string<AttributeList>
	{
		static std::string value(const AttributeList& v, StringMode a_Mode = DEFAULT)
		{
			std::ostringstream strm;
			object_rawptr_set printed;
			v.str( strm, printed, a_Mode );
			return strm.str();
		}
	};
}

}

#endif
















/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
