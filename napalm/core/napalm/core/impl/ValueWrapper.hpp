#ifndef _NAPALM_VALUEWRAPPER__H_
#define _NAPALM_VALUEWRAPPER__H_

#include "../meta.hpp"
#include "../Attribute.h"
#include <vector>


namespace napalm {

	template<class V>
	class Wrapper;

	namespace detail {

		bool ValueWrapperLessThan(const Object& a, const Object& b);
		bool ValueWrapperLessThan(const Attribute& a, const Attribute& b);

		bool ValueWrapperEqual(const Object& a, const Object& b);
		bool ValueWrapperEqual(const Attribute& a, const Attribute& b);

	}


	// Wrapper is too generic as a name
	template<class V>
	class Wrapper
	{
	public:
		template<typename T>
		Wrapper( const T& a_Value, typename boost::enable_if<is_napalm_attrib_type<T>, void* >::type dummy = 0 )
			: m_Ptr( new TypedAttribute<T>(a_Value) )
		{
		}


		// Automatically convert vectors not using counted_allocator
		template< typename S, typename Q >
		Wrapper( const std::vector<S, Q> & a_Value,
				typename boost::enable_if<is_napalm_attrib_type< std::vector< S, util::counted_allocator<S> > >, void* >::type dummy = 0 )
				: m_Ptr( new TypedAttribute< std::vector<S, util::counted_allocator<S> > >() )
		{
			TypedAttribute< std::vector<S, util::counted_allocator<S> > > * attr =
					static_cast< TypedAttribute< std::vector<S, util::counted_allocator<S> > > *>( m_Ptr.get());
			attr->value().reserve( a_Value.size() );
			for( typename std::vector< S, Q >::const_iterator it = a_Value.begin(); it != a_Value.end(); ++it )
			{
				attr->value().push_back( *it );
			}
		}

		// Automatically convert sets not using counted_allocator or util::less
		template< typename S, typename Q, typename R >
		Wrapper( const std::set<S, Q, R> & a_Value,
				typename boost::enable_if<is_napalm_attrib_type< std::set<S, util::less<S>, util::counted_allocator<S> > >, void* >::type dummy = 0 )
				: m_Ptr( new TypedAttribute<std::set<S, util::less<S>, util::counted_allocator<S> > >() )
		{
			TypedAttribute<std::set<S, util::less<S>, util::counted_allocator<S> > > * attr =
					static_cast< TypedAttribute<std::set<S, util::less<S>, util::counted_allocator<S> > >* >( m_Ptr.get() );
			for( typename std::set<S, Q, R>::const_iterator it = a_Value.begin(); it != a_Value.end(); ++it )
			{
				attr->value().insert( *it );
			}
		}

		inline Wrapper( const char * a_Value )
			: m_Ptr( new TypedAttribute<std::string>(a_Value) )
		{
		}

		inline Wrapper( boost::shared_ptr<V> a_Ptr )
			: m_Ptr( a_Ptr )
		{
		}

		inline Wrapper()
		{
		}

		void operator=(Wrapper & aw)
		{
			m_Ptr = aw.m_Ptr;
		}

		bool isNull() const
		{
			return m_Ptr.get() == NULL;
		}

		boost::enable_if<boost::is_same<V, Attribute>,bool>
		operator<(const Wrapper & rhs ) const
		{
			return detail::ValueWrapperLessThan(*m_Ptr, *(rhs.m_Ptr));
		}

		boost::enable_if<boost::is_same<V, Attribute>,bool>
		operator==(const Wrapper & rhs ) const
		{
			return detail::ValueWrapperEqual(*m_Ptr, *(rhs.m_Ptr));
		}

		boost::shared_ptr<V> get()
		{
			return m_Ptr;
		}

		const boost::shared_ptr<V> & operator->() const
		{
			return m_Ptr;
		}

		const boost::shared_ptr<V> get() const
		{
			return m_Ptr;
		}

		const V & operator*() const
		{
			return *m_Ptr;
		}

	protected:

		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version);

	//private:
		boost::shared_ptr<V> m_Ptr;
	};


///////////////////////// impl

template<class V>
template<class Archive>
void Wrapper<V>::serialize(Archive & ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("value", m_Ptr);
}


}

template< typename V>
std::ostream& operator <<(std::ostream& os, const napalm::Wrapper<V>& o)
{
	return os << *o;
}

#endif











