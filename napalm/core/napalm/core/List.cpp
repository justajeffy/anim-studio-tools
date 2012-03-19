
#include <boost/static_assert.hpp>
#include <pystring.h>
#include "Dispatcher.h"
#include "List.h"


namespace napalm {

	bool AttributeList::operator<(const AttributeList& rhs) const
	{
		const_iterator it = m_contents.begin();
		const_iterator it2 = rhs.m_contents.begin();

		for(; it != m_contents.end() && it2 != rhs.m_contents.end(); ++it, ++it2)
		{
			Dispatcher::Comparison c = Dispatcher::instance().attribCompare(*(*it), *(*it2));
			if(c == Dispatcher::LESS_THAN)
				return true;
			else if(c == Dispatcher::GREATER_THAN)
				return false;
		}

		return it2 != rhs.m_contents.end();
	}


	bool AttributeList::operator==(const AttributeList& rhs) const
	{
		if( m_contents.size() != rhs.m_contents.size() )
			return false;

		const_iterator it = m_contents.begin();
		const_iterator it2 = rhs.m_contents.begin();
		for( ; it != m_contents.end(); ++it, ++it2 )
		{
			if(!Dispatcher::instance().attribEqual(*(*it), *(*it2)))
				return false;
		}
		return true;
	}


	std::ostream& AttributeList::dump(std::ostream& os, object_rawptr_set& printed) const
	{
		return str(os, printed);
	}



	///////////////////////// impl

	AttributeList* AttributeList::clone(object_clone_map& cloned) const
	{
		AttributeList * rv( new AttributeList() );
		for( const_iterator it = m_contents.begin(); it != m_contents.end(); ++it )
		{
			rv->push_back( *it );
		}
		return rv;
	}

	std::ostream& AttributeList::str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type) const
	{
		os << '(';
		bool first = true;
		for(const_iterator it = m_contents.begin(); it!=m_contents.end(); ++it, first=false)
		{
			if(!first)
				os << ", ";
			(*it)->str( os, printed, a_Type );
		}

		os << ')';
		return os;
	}
}
