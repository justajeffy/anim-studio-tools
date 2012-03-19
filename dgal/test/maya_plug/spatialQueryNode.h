#ifndef _INVERSE_MATRIX_H
#define _INVERSE_MATRIX_H

#include <maya/MPlug.h>
#include <maya/MPxNode.h>
#include "mayaAdaptor.hpp"
#include "spatials/octree.hpp"
#include "spatials/pointSet.hpp"
#include "spatials/polygonSet.hpp"

class SpatialQuery: public MPxNode
{
public :
	static  void*		sCreateNode();
	static  MStatus		sInitialize();

	SpatialQuery();
	~SpatialQuery();

	virtual MStatus  	compute ( const  MPlug & plug,  MDataBlock & dataBlock );

	static  MObject		attrInMesh1;
	static  MObject     attrInMesh;
	static  MObject     attrOutMesh;
	static  MObject 	attrBoxTransform;

	static  MTypeId		id;

	int 	_nPt,_nPt1;
	dgal::Octree< dgal::PolygonSet<MObject> >* _pOct;
	dgal::PolygonSet<MObject> *_pPoSet;
	dgal::PolygonSet<MObject> *_pPoSet1;
};
#endif
