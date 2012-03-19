#ifndef _DGAL_KDTREE__H_
#define _DGAL_KDTREE__H_

#include <OpenEXR/ImathVec.h>
#include <vector>
#include "../adaptors/points.hpp"

namespace dgal
{
//-----------------------------------------------------------------------------
	/*
	 * @class Octree
	 * @brief
	 * acceleration structure for spatial queries on generic dataSet..
	 */
	template< class Storage >
	class KDTree
	{
	public:
		typedef points_adaptor< Storage > Adaptor;
		typedef std::vector< unsigned int > Index;
		typedef typename std::vector< unsigned int >::iterator IndexIterator;

		KDTree( const Storage & data, int maxLeaf = 10 );
		~KDTree();

		void query(	const Imath::Box< typename Adaptor::elem_type > & qBox,
			Index & outResult ) const;

		void query(	const typename points_adaptor< Storage >::elem_type & point, 
			typename points_adaptor< Storage >::scalar radius,
			Index & outResult ) const;

		class Node;
		const Node * getTree() const { return m_tree; }

	private:
		struct Sorter;

		Adaptor m_adaptor;
		int m_maxLeaf;
		Index m_index;
		Node * m_tree;
	};

//-----------------------------------------------------------------------------
	// implementation
	template< class Storage >
	KDTree< Storage >::KDTree( const Storage & data, int maxLeaf )
		:	m_adaptor( data )
		,	m_maxLeaf( maxLeaf )
		,	m_tree( new Node() )
	{
		m_index.resize( static_cast< unsigned int >( m_adaptor.size() ) );
		for ( unsigned int i = 0 ; i < m_adaptor.size() ; ++i )
			m_index[ i ] = i;

		m_tree->build( m_adaptor, m_index.begin(), m_index.end(), m_maxLeaf );
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	KDTree< Storage >::~KDTree()
	{
		delete m_tree;
		m_tree = NULL;
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	void 
		KDTree< Storage >::query( const Imath::Box< typename Adaptor::elem_type > & qBox,
		Index & outResult ) const
	{
		assert( m_tree );
		m_tree->query( m_adaptor, qBox, outResult );
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	void 
		KDTree< Storage >::query( const typename points_adaptor< Storage >::elem_type & point, 
		typename points_adaptor< Storage >::scalar radius,
		Index & outResult ) const
	{
		assert( m_tree );
		// use the squared of the radius hereon down.
		m_tree->query( m_adaptor, point, radius, radius*radius, outResult );
	}


//-----------------------------------------------------------------------------
	// Node definition
	template< class Storage >
	class KDTree< Storage >::Node
	{
	public:
		typedef typename Imath::Box< typename KDTree< Storage >::Adaptor::elem_type > BBox;

		Node();
		void build( const Adaptor & a, IndexIterator f, IndexIterator l, int mL );
		~Node();
		void query(	
			const Adaptor & a,
			const BBox & qBox,
			typename KDTree< Storage >::Index & outResult ) const;

		void query( 
			const Adaptor & a,
			const typename points_adaptor< Storage >::elem_type & point, 
			typename points_adaptor< Storage >::scalar radius,
			typename points_adaptor< Storage >::scalar radiusSqr,
			Index & outResult ) const;

		bool isLeaf() const { return m_axis == -1; }
		const Node & getBefore() const { return m_children[0]; }
		const Node & getAfter() const { return m_children[1]; }
		const BBox & getBBox() const { return m_bbox; }
		const typename KDTree< Storage >::IndexIterator & getFirst() const { return m_first; }
		const typename KDTree< Storage >::IndexIterator & getMid() const { return m_mid; }
		const typename KDTree< Storage >::IndexIterator & getLast() const { return m_last; }
		const typename KDTree< Storage >::Adaptor & getAdaptor() const { return m_adaptor; }
		unsigned int getAxis() const { return m_axis; }
	private:
		std::vector< Node > m_children;
		unsigned int m_axis;
		BBox m_bbox;
		typename KDTree< Storage >::IndexIterator m_first, m_last, m_mid;
		int m_maxLeaf;
	};

//-----------------------------------------------------------------------------
	template< class Storage >
	KDTree< Storage >::Node::Node()
		:	m_axis( -1 )
	{
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	void 
		KDTree< Storage >::Node::build( const Adaptor & a, IndexIterator f, IndexIterator l, int mL )
	{
		m_first = f;
		m_last = l;
		m_maxLeaf = mL;

		for ( IndexIterator it = m_first ; it != m_last ; ++it )
			m_bbox.extendBy( a[*it] );

		const size_t t = m_last - m_first;
		if ( t <= m_maxLeaf )
			return;

		m_mid = ( m_last - m_first) / 2 + m_first;

		m_axis = m_bbox.majorAxis();
		std::nth_element( m_first, m_mid, m_last, Sorter( a, m_axis ) );

		m_children.resize( 2 );
		m_children[0].build( a, m_first, m_mid, m_maxLeaf );
		m_children[1].build( a, m_mid+1, m_last, m_maxLeaf );
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	KDTree< Storage >::Node::~Node()
	{
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	void 
		KDTree< Storage >::Node::query(	
		const Adaptor & a,
		const BBox & qBox,
		typename KDTree< Storage >::Index & outResult ) const
	{
		if ( isLeaf() )
		{
			// pre-empt worse-case scenario
			outResult.reserve( outResult.size() + (m_last-m_first) );
			for ( typename KDTree< Storage >::IndexIterator it = m_first ;
				it != m_last ; ++it )
			{
				if ( qBox.intersects( a[ *it ] ) )
					outResult.push_back( *it );
			}
		}
		else
		{
			assert( m_children.size() == 2 );
			bool afterMin = false;
			bool beforeMax = false;
			const typename Adaptor::scalar mid_v = a[ *m_mid ][ m_axis ];
			if ( qBox.min[ m_axis ] < mid_v )
			{
				m_children[0].query( a, qBox, outResult );
				afterMin = true;
			}
			if ( qBox.max[ m_axis ] > mid_v )
			{
				m_children[1].query( a, qBox, outResult );
				beforeMax = true;
			}

			if ( afterMin && beforeMax && qBox.intersects( a[ *m_mid ] ) )
				outResult.push_back( *m_mid );
		}
	}

//-----------------------------------------------------------------------------
	template< class Storage >
	void 
		KDTree< Storage >::Node::query( 
		const Adaptor & a,
		const typename points_adaptor< Storage >::elem_type & point, 
		typename points_adaptor< Storage >::scalar radius,
		typename points_adaptor< Storage >::scalar radiusSqr,
		Index & outResult ) const
	{
		if ( isLeaf() )
		{
			// pre-empt worse-case scenario
			outResult.reserve( outResult.size() + (m_last-m_first) );
			for ( typename KDTree< Storage >::IndexIterator it = m_first ;
				it != m_last ; ++it )
			{
				if ( ( point - a[ *it ] ).length2() < radiusSqr )
					outResult.push_back( *it );
			}
		}
		else
		{
			assert( m_children.size() == 2 );
			bool afterMin = false;
			bool beforeMax = false;
			const typename Adaptor::scalar mid_v = a[ *m_mid ][ m_axis ];
			if ( ( point[ m_axis ] - radius ) < mid_v )
			{
				m_children[0].query( a, point, radius, radiusSqr, outResult );
				afterMin = true;
			}
			if ( ( point[ m_axis ] + radius ) > mid_v )
			{
				m_children[1].query( a, point, radius, radiusSqr, outResult );
				beforeMax = true;
			}

			if ( afterMin && beforeMax && ( point - a[ *m_mid ] ).length2() < radiusSqr )
				outResult.push_back( *m_mid );
		}
	}


//-----------------------------------------------------------------------------
	// Sorting definition
	template< class Storage >
	struct KDTree< Storage >::Sorter
	{
		Sorter( const typename KDTree< Storage >::Adaptor & a_adaptor, unsigned int a_axis ) 
			: m_adaptor( a_adaptor ), m_axis( a_axis ){}
		bool operator() ( unsigned int i, unsigned int j )
		{
			return m_adaptor[i][m_axis] < m_adaptor[j][m_axis];
		}
		const typename KDTree< Storage >::Adaptor m_adaptor;
		const unsigned int m_axis;
	};

} // namespace dgal

#endif // _DGAL_KDTREE__H_


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
