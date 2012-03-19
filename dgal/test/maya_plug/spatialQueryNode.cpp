#include "spatialQueryNode.h"
#include <maya/MFnCompoundAttribute.h>
#include <maya/MDataBlock.h>
#include <maya/MMatrix.h>
#include<maya/MFnMatrixAttribute.h>
#include<maya/MFnTypedAttribute.h>
#include<maya/MFnMesh.h>
#include<maya/MFnMeshData.h>
#include<maya/MFloatPointArray.h>
#include<maya/MPointArray.h>
#include<maya/MFnPlugin.h>
#include <assert.h>
#include<maya/MBoundingBox.h>


MTypeId     SpatialQuery::id(0x00113200);

MObject		SpatialQuery::attrInMesh1;
MObject		SpatialQuery::attrInMesh;
MObject		SpatialQuery::attrOutMesh;
MObject		SpatialQuery::attrBoxTransform;

void* SpatialQuery::sCreateNode()
{
	return new SpatialQuery();
}


SpatialQuery::SpatialQuery()
{
	_pOct = NULL;
	_pPoSet =NULL;
	_nPt = 0;
	_nPt1 =0;
}

SpatialQuery::~SpatialQuery()
{
	if (_pOct !=NULL)
		delete _pOct;
	if (_pPoSet != NULL)
		delete _pPoSet;
}

MStatus SpatialQuery::sInitialize()
{
	MStatus stat;
	MFnTypedAttribute tAttr;

	//attrInMesh
	attrInMesh= tAttr.create("inMesh","im", MFnData::kMesh);
	tAttr.setReadable(true);
	tAttr.setWritable(true);
	stat = addAttribute( attrInMesh );
	if (!stat)
	{
		stat.perror(" addAttribute( attrInMesh ); ");
		return stat;
	}

	//attrInMesh
	attrInMesh1= tAttr.create("inMesh1","im1", MFnData::kMesh);
	tAttr.setReadable(true);
	tAttr.setWritable(true);
	stat = addAttribute( attrInMesh1 );
	if (!stat)
	{
		stat.perror(" addAttribute( attrInMesh ); ");
		return stat;
	}

	//attrOutMesh
	attrOutMesh = tAttr.create("outMesh","om", MFnData::kMesh);
	tAttr.setReadable(true);
	tAttr.setWritable(false);
	stat = addAttribute( attrOutMesh );
	if (!stat)
	{
		stat.perror(" addAttribute( attrOutMesh ); ");
		return stat;
	}

	//attrBoxTransform
	MFnMatrixAttribute  fn_Tr_attr;
	attrBoxTransform=fn_Tr_attr.create("boxTransform", "bt");
	fn_Tr_attr.setWritable(true);
	fn_Tr_attr.setConnectable(true);
	stat = addAttribute(attrBoxTransform);
	if (!stat)
	{
		stat.perror("status = addAttribute(attrBoxTransform)");
	}

	//affects
	attributeAffects(attrInMesh1, attrOutMesh);
	attributeAffects(attrInMesh, attrOutMesh);
	attributeAffects(attrBoxTransform, attrOutMesh);

	return MStatus::kSuccess;
}


void draw(const dgal::Cube< Imath::Vec3<float> >& cub)
{
	Imath::Vec3<float> pos = cub.min();
	float size = cub.size();

	MPointArray pa;
	MIntArray faceCounts;
	MIntArray faceConnects;
	MVectorArray normals;

	MStatus ms;
	int i;
	const int num_faces                     = 6;
	const int num_face_connects     = 24;
	const double normal_value   = 0.5775;
	const int uv_count                      = 14;

	pa.clear(); faceCounts.clear(); faceConnects.clear();


	pa.append( MPoint( pos.x, pos.y, pos.z ));
	pa.append( MPoint(  size+pos.x, pos.y, pos.z));
	pa.append( MPoint(  size+pos.x, pos.y,  size +pos.z ));
	pa.append( MPoint(  pos.x, pos.y,  size +pos.z ));
	pa.append( MPoint( pos.x,  size+pos.y, pos.z ));
	pa.append( MPoint( pos.x,  size+pos.y,  size +pos.z));
	pa.append( MPoint(  size+pos.x,  size+pos.y,  size +pos.z));
	pa.append( MPoint(  size+pos.x,  size+pos.y, pos.z));

	normals.append( MVector( -normal_value, -normal_value, -normal_value ) );
	normals.append( MVector(  normal_value, -normal_value, -normal_value ) );
	normals.append( MVector(  normal_value, -normal_value,  normal_value ) );
	normals.append( MVector( -normal_value, -normal_value,  normal_value ) );
	normals.append( MVector( -normal_value,  normal_value, -normal_value ) );
	normals.append( MVector( -normal_value,  normal_value,  normal_value ) );
	normals.append( MVector(  normal_value,  normal_value,  normal_value ) );
	normals.append( MVector(  normal_value,  normal_value, -normal_value ) );


	// Set up an array containing the number of vertices
	// for each of the 6 cube faces (4 verticies per face)
	//
	int face_counts[num_faces] = { 4, 4, 4, 4, 4, 4 };

	for ( i=0; i<num_faces; i++ )
	{
			faceCounts.append( face_counts[i] );
	}

	// Set up and array to assign vertices from pa to each face
	//
	int face_connects[ num_face_connects ] = {      0, 1, 2, 3,
													4, 5, 6, 7,
													3, 2, 6, 5,
													0, 3, 5, 4,
													0, 4, 7, 1,
													1, 7, 6, 2      };
	for ( i=0; i<num_face_connects; i++ )
	{
			faceConnects.append( face_connects[i] );
	}

	//Creator
	MFnMesh creator ;
	 MObject resObj = creator.create(pa.length(),faceCounts.length(),pa,faceCounts,faceConnects,MObject::kNullObj,&ms);
	if (!ms)
	{
		ms.perror("Cant' build the Cell Cube ...");
	}

}

void draw(const Imath::Box< Imath::Vec3<float> >& cub)
{
	Imath::Vec3<float> pos = cub.min;
	Imath::Vec3<float> size = cub.size();


	MPointArray pa;
	MIntArray faceCounts;
	MIntArray faceConnects;
	MVectorArray normals;

	MStatus ms;
	int i;
	const int num_faces                     = 6;
	const int num_face_connects     = 24;
	const double normal_value   = 0.5775;
	const int uv_count                      = 14;

	pa.clear(); faceCounts.clear(); faceConnects.clear();


	pa.append( MPoint( pos.x, pos.y, pos.z ));
	pa.append( MPoint(  size.x+pos.x, pos.y, pos.z));
	pa.append( MPoint(  size.x+pos.x, pos.y,  size.z +pos.z ));
	pa.append( MPoint(  pos.x, pos.y,  size.z +pos.z ));
	pa.append( MPoint( pos.x,  size.y+pos.y, pos.z ));
	pa.append( MPoint( pos.x,  size.y+pos.y,  size.z +pos.z));
	pa.append( MPoint(  size.x+pos.x,  size.y+pos.y,  size.z +pos.z));
	pa.append( MPoint(  size.x+pos.x,  size.y+pos.y, pos.z));

	normals.append( MVector( -normal_value, -normal_value, -normal_value ) );
	normals.append( MVector(  normal_value, -normal_value, -normal_value ) );
	normals.append( MVector(  normal_value, -normal_value,  normal_value ) );
	normals.append( MVector( -normal_value, -normal_value,  normal_value ) );
	normals.append( MVector( -normal_value,  normal_value, -normal_value ) );
	normals.append( MVector( -normal_value,  normal_value,  normal_value ) );
	normals.append( MVector(  normal_value,  normal_value,  normal_value ) );
	normals.append( MVector(  normal_value,  normal_value, -normal_value ) );


	// Set up an array containing the number of vertices
	// for each of the 6 cube faces (4 verticies per face)
	//
	int face_counts[num_faces] = { 4, 4, 4, 4, 4, 4 };

	for ( i=0; i<num_faces; i++ )
	{
			faceCounts.append( face_counts[i] );
	}

	// Set up and array to assign vertices from pa to each face
	//
	int face_connects[ num_face_connects ] = {      0, 1, 2, 3,
													4, 5, 6, 7,
													3, 2, 6, 5,
													0, 3, 5, 4,
													0, 4, 7, 1,
													1, 7, 6, 2      };
	for ( i=0; i<num_face_connects; i++ )
	{
			faceConnects.append( face_connects[i] );
	}

	//Creator
	MFnMesh creator ;
	 MObject resObj = creator.create(pa.length(),faceCounts.length(),pa,faceCounts,faceConnects,MObject::kNullObj,&ms);
	if (!ms)
	{
		ms.perror("Cant' build the Cell Cube ...");
	}

}

MStatus  SpatialQuery::compute ( const  MPlug & plug,  MDataBlock & dataBlock )
{
	MStatus stat;
	if ( plug == attrOutMesh)
	{

		// get the Box/plane Transform
		// attrBoxTransform
		MDataHandle inmaH = dataBlock.inputValue(attrBoxTransform,&stat);
		assert(stat == MStatus::kSuccess);
		if (stat != MS::kSuccess)
		{
			stat.perror(" dataBlock.inputValue(attrInMatrix,&stat)");
			return stat;
		}
		MMatrix boxTr = inmaH.asMatrix();
		//

		// get the input mesh
		// get attrInMesh
		MDataHandle inMeshObjH = dataBlock.inputValue(attrInMesh,&stat);
		if (!stat)
		{
			stat.perror(" data.outputValue(attrInMesh,&stat)");
			return stat;
		}
		MObject inMeshObj =inMeshObjH.asMesh();
		MFnMesh inMeshFn(inMeshObj,&stat);
		if (! stat)
		{
			stat.perror("inMeshFn(inMeshObj,&stat");
			return stat;
		}
		/*
		//get attrInMesh1
		MDataHandle inMeshObjH1 = dataBlock.inputValue(attrInMesh1,&stat);
		if (!stat)
		{
			stat.perror(" data.outputValue(attrInMesh1,&stat)");
			return stat;
		}
		MObject inMeshObj1 =inMeshObjH1.asMesh();
		MFnMesh inMeshFn1(inMeshObj1,&stat);
		if (! stat)
		{
			stat.perror("inMeshFn1(inMeshObj1,&stat");
			return stat;
		}
		*/
		//


		//get mesh indices
		MIntArray aPolygonConnects[2];
		MIntArray aPolyCount[2];
		stat = inMeshFn.getVertices(aPolyCount[0],aPolygonConnects[0]);
		if (!stat)
		{
			stat.perror("inMeshFn.getVertices");
			return stat;
		}
	/*	stat = inMeshFn1.getVertices(aPolyCount[1],aPolygonConnects[1]);
		if (!stat)
		{
			stat.perror("inMeshFn.getVertices");
			return stat;
		}*/

		//get mesh verts
		MFloatPointArray aPoints[2];
		stat = inMeshFn.getPoints(aPoints[0],MSpace::kObject);
		if (!stat)
		{
			stat.perror("inMeshFn.getVertices");
			return stat;
		}
		/*
		stat = inMeshFn1.getPoints(aPoints[1],MSpace::kObject);
		if (!stat)
		{
			stat.perror("inMeshFn.getVertices");
			return stat;
		}
*/
		//--- outputGeometry Handle ---
		MDataHandle outGeomH =  dataBlock.outputValue(attrOutMesh,&stat);
		if (!stat)
		{

			stat.perror(" data.outputValue(attrOutGeometry,&stat)");
			return stat;
		}

		if (aPoints[0].length()!= _nPt )//|| aPoints[1].length()!= _nPt1)
		{
			std::cout<<"REBUILDING THE OCTREE"<<std::endl;
			if (_pOct !=NULL)
				delete _pOct;
			if (_pPoSet != NULL)
				delete _pPoSet;

			// *****   Build the polygonSet *****
			// *****   Build the octree  	*****
			//dgal::PointSet<MFloatPointArray> ps(points);
			std::set<unsigned int> components;
			for (unsigned c=0;c<aPolyCount[0].length()/3;c++)
			{
				components.insert(c);
			}
			std::set<unsigned int> components1;
			for (unsigned c=0;c<aPolyCount[1].length()/2;c++)
			{
				components1.insert(c);
			}
			_pPoSet = 	new dgal::PolygonSet< MObject >(inMeshObj);
			_pOct = 	new dgal::Octree< dgal::PolygonSet<MObject> >(*_pPoSet,15);//,&components);
			dgal::Cube< Imath::Vec3<float> > octreeTransform;
		//	_pOct->getAffineTransform(octreeTransform);
		//	_pPoSet1 = 	new dgal::PolygonSet< MObject >(inMeshObj1,&octreeTransform);//,&components);
		//	if (inMeshObj1.isNull()==false)
		//		_pOct->add(*_pPoSet1,&components1);

			_nPt = aPoints[0].length();
			//_nPt1 = aPoints[1].length();
		}

		//transform the unit bbox
		MPoint pQMin(0.0f,0.0f,0.0f);
		MPoint pQMax(1.0f,1.0f,1.0f);
		MPoint pQMinTr = pQMin*boxTr;
		MPoint pQMaxTr = pQMax*boxTr;

		std::vector<std::set<unsigned> > aIntersected;
		std::vector<std::set<unsigned> > aLeft;
		std::vector<std::set<unsigned> > aRight;
		Imath::Vec3<float> qMinTr(pQMinTr.x, pQMinTr.y, pQMinTr.z);
		Imath::Vec3<float> qMaxTr(pQMaxTr.x, pQMaxTr.y, pQMaxTr.z);
		Imath::Box< Imath::Vec3<float> > qBox(qMinTr,qMaxTr);

		//Transform the query plane
		MVector Yaxis(0.0f,1.0f,0.0f);
		MPoint orig(0.0f,0.0f,0.0f);
		MVector YaxisTr = Yaxis*boxTr;
		YaxisTr.normalize();
		MPoint origTr = orig*boxTr;

		Imath::Vec3<float> N( YaxisTr.x, YaxisTr.y, YaxisTr.z );
		Imath::Plane3<float> plane(N,MVector(origTr)*YaxisTr);


		//dgal::Cube< float > testCube(qMinTr,qMaxTr);
		//draw(testCube);
		dgal::Cube< Imath::Vec3<float> > testCube;
		//_pOct->query(qBox,aIntersected);

		// *****   Build Perform the plane query  *****
		short axis = 1;
		std::vector<Imath::Box<Imath::Vec3<float> > > aBox;
		_pOct->query(Imath::Vec3<float>(origTr.x,origTr.y,origTr.z),axis,false,aLeft);
		//_pOct->query(qBox,aLeft);
		//_pOct->query(plane,&aLeft);

		for(unsigned i=0;i<aBox.size();i++)
		{
			draw(aBox[i]);
		}
		// draw the query region
		std::vector<dgal::Cube< Imath::Vec3<float> > > ac;
		bool bIntersect = _pOct->intersect(qBox,1e-4f,&ac);
		std::cout<<"intersect = "<<bIntersect<<std::endl;
		if (ac.size()>0);
			//draw(ac[0]);


		//build resulting mesh
		std::vector<std::set<unsigned> >::const_iterator vcIt;
		std::set<unsigned>::const_iterator cIt;

		MIntArray aStart[2];

		unsigned meshIndex =0;
		for (vcIt =  aLeft.begin(); vcIt != aLeft.end(); vcIt++, meshIndex++)
		{

			int cumulate =0;
			for (unsigned i=0;i<aPolyCount[meshIndex].length(); i++ )
			{
				aStart[meshIndex].append(cumulate);
				cumulate+=aPolyCount[meshIndex][i];
			}
		}
		//build out maya mesh
		MIntArray outPolygonConnects;
		MIntArray outPolyCount;
		MPointArray outPts;


		meshIndex =0;
		for (vcIt =  aLeft.begin(); vcIt != aLeft.end(); vcIt++, meshIndex++)
		{

			const std::set<unsigned>& indSet = *vcIt;
			for (cIt =  indSet.begin(); cIt != indSet.end(); cIt++)
			{
				unsigned index = *cIt;

				assert(index < aPolyCount[meshIndex].length());
				assert(index < aStart[meshIndex].length());

				unsigned nVerts = aPolyCount[meshIndex][index ];
				unsigned start = aStart[meshIndex][index];
				//std::cout<< *cIt << "  ";
				for (unsigned iv =start; iv < nVerts+start; iv++)
				{
					unsigned v = aPolygonConnects[meshIndex][iv];
					assert(v<aPoints[meshIndex].length());
					outPolygonConnects.append(outPts.length());
					outPts.append(aPoints[meshIndex][v]);
				}
				outPolyCount.append(nVerts);
			}
		}
		std::cout<<std::endl;

		/*
		std::vector< dgal::Cube<float> > cubes;
		oct.getCubes(true,-1,cubes); // all levels
		for (uint i=0;i<cubes.size();i++)
		{
			draw(cubes[i]);
		}
		*/
		unsigned mi, ma;
		_pOct->getStat(mi,ma);
		std::cout<<"nim Min elem by cell = "<<mi<<std::endl;
		std::cout<<"nim Max elem by cell = "<<ma<<std::endl;

		//

		MFnMeshData dataCreator;
		MObject newOutputData = dataCreator.create(&stat);
		if (!stat) { stat.perror("SpatialQuery : dataCreator.create"); return stat;}

		//---- create the output mesh Copy from input ----
		MFnMesh creator;
		MObject newMesh;

		if (outPolyCount.length()>0)
		{
			std::cout<<"outPolyCount.length()"<<outPolyCount.length()<<std::endl;
			std::cout<<"outPts.length()"<<outPts.length()<<std::endl;
			std::cout<<"outPolygonConnects.length()"<<outPolygonConnects.length()<<std::endl;
			newMesh = creator.create(outPts.length(),outPolyCount.length(),outPts,outPolyCount,outPolygonConnects, newOutputData,&stat );
			if (!stat)
			{
				stat.perror("SpatialQuery : creator.create"); return stat;
			}
		}
		else
		{
		///
			MObject copyObj = inMeshFn.copy(inMeshObj,newOutputData);
		}

		// set the output an make the plug clean
		stat = outGeomH.set( newOutputData );
		if (!stat) {stat.perror("SpatialQuery : outGeomH.set \n"); return stat;}

		stat = dataBlock.setClean( plug );
		if (!stat) {stat.perror("SpatialQuery : data.setClean \n"); return stat;}
		return MStatus::kSuccess;

	}
	else
	{
		return MS::kFailure;
	}

}

MStatus initializePlugin( MObject obj )
{
	MStatus status;
	MFnPlugin plugin(obj, "Drd studios", "5.0", "Any");



	status = plugin.registerNode(	"spatialQuery",
									SpatialQuery::id,
									SpatialQuery::sCreateNode,
									SpatialQuery::sInitialize,
									MPxNode::kDependNode    );
}

MStatus uninitializePlugin( MObject obj)
{
	MStatus status;
	MFnPlugin plugin( obj );


	status = plugin.deregisterNode(SpatialQuery::id);
}
