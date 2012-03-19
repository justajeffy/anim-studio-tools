#ifndef _NAPALM_TYPEDATTRIBUTE__H_
#define _NAPALM_TYPEDATTRIBUTE__H_

#include "meta.hpp"
#include "Attribute.h"
#include "util/default_construct.hpp"
#include "util/to_string.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/assert.hpp>


namespace napalm {


	/*
	 * @class TypedAttribute
	 * @brief
	 * Napalm attribute class templatised on value type.
	 */
	template<typename T>
	class TypedAttribute : public Attribute
	{
		BOOST_MPL_ASSERT((is_napalm_attrib_type<T>));

	public:

		typedef T value_type;

		TypedAttribute(const T& value = util::default_construction<T>::value())
			: m_value ( value )
		{
		}

		template<typename S>
		TypedAttribute(const TypedAttribute<S>& rhs)
		: m_value(rhs.m_value){}

		// 'tuple' style vector & set constructors
		// (aj) overkill I reckon..?
		template<typename S>
		TypedAttribute(const S & A0, const S & A1 );
		template<typename S>
		TypedAttribute(const S & A0, const S & A1, const S & A2 );
		template<typename S>
		TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3 );
		template<typename S>
		TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4 );
		template<typename S>
		TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4,
					   const S & A5 );
		template<typename S>
		TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4,
					   const S & A5, const S & A6 );
		template<typename S>
		TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4,
					   const S & A5, const S & A6, const S & A7 );

		template< typename S, typename Q >
		TypedAttribute( const std::vector<S, Q> & value,
				typename boost::enable_if<boost::is_same<T, std::vector<S, util::counted_allocator<S> > >, void* >::type dummy = 0 )
		{
			m_value.reserve( value.size() );
			for( typename std::vector<S, Q>::const_iterator it = value.begin(); it != value.end(); ++it )
			{
				m_value.push_back( *it );
			}
		}

		template< typename S, typename Q, typename R >
		TypedAttribute( const std::set<S, Q, R> & value,
				typename boost::enable_if<boost::is_same<T, std::set<S, util::less<S>, util::counted_allocator<S> > >, void* >::type dummy = 0 )
		{
			for( typename std::set<S, Q, R>::const_iterator it = value.begin(); it != value.end(); ++it )
			{
				m_value.insert( *it );
			}
		}

		template<typename S>
		typename boost::enable_if<boost::is_same<T, std::vector<S, util::counted_allocator<S> > >, void>::type
		Append( const S & value )
		{
			m_value.push_back( value );
		}

		template<typename S>
		typename boost::enable_if<boost::is_same<T, std::set<S, util::less<S>, util::counted_allocator<S> > >, void>::type
		Append( const S & value )
		{
			m_value.insert( value );
		}

		virtual ~TypedAttribute(){}

		const T& value() const { return m_value; }

		T& value() { return m_value; }

		virtual object_ptr clone(object_clone_map& cloned) const;

		virtual std::ostream& str(std::ostream& os, object_rawptr_set& printed,
			util::StringMode a_Type = util::DEFAULT) const;

		virtual const std::type_info& type() const {
			return typeid(T);
		}

		// why? (aj) can just go typeid(This::value_type) instead.
		static const std::type_info & Type()
		{
			return typeid(T);
		}

	protected:

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version);

	protected:

		T m_value;
	};


///////////////////////// impl

template<typename T>
object_ptr TypedAttribute<T>::clone(object_clone_map& cloned) const
{
	assert(cloned.find(this) == cloned.end());
	boost::shared_ptr<TypedAttribute> pclone(new TypedAttribute(*this));
	cloned.insert(object_clone_map::value_type(this, pclone));

	return pclone;
}


template<typename T>
std::ostream& TypedAttribute<T>::str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type ) const
{
	os << util::to_string<T>::value(m_value, a_Type);
	return os;
}


template<typename T>
template<class Archive>
void TypedAttribute<T>::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("base_class", base_object<Attribute>(*this));
	ar & make_nvp("value", m_value);
}

template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1 )
{
	Append( A0 );
	Append( A1 );
}

template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1, const S & A2 )
{
	Append( A0 );
	Append( A1 );
	Append( A2 );
}

template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3 )
{
	Append( A0 );
	Append( A1 );
	Append( A2 );
	Append( A3 );
}


template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4 )
{
	Append( A0 );
	Append( A1 );
	Append( A2 );
	Append( A3 );
	Append( A4 );
}

template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4,
								  const S & A5)
{
	Append( A0 );
	Append( A1 );
	Append( A2 );
	Append( A3 );
	Append( A4 );
	Append( A5 );
}

template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4,
								  const S & A5, const S & A6)
{
	Append( A0 );
	Append( A1 );
	Append( A2 );
	Append( A3 );
	Append( A4 );
	Append( A5 );
	Append( A6 );
}


template<typename T>
template<typename S>
TypedAttribute<T>::TypedAttribute(const S & A0, const S & A1, const S & A2, const S & A3, const S & A4,
								  const S & A5, const S & A6, const S & A7)
{
	Append( A0 );
	Append( A1 );
	Append( A2 );
	Append( A3 );
	Append( A4 );
	Append( A5 );
	Append( A6 );
	Append( A7 );
}

} // ns


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
