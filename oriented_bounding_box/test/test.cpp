#include "../src/OrientedBoundingBox.h"

#include <vector>
#define STEPS 8

int main(int argc,char** argv)
{
	typedef double scalar_type;
	for( uint32_t i = 0; i < 16; ++i )
	{
		std::vector<Imath::Vec3<scalar_type> > buffer(8);
		int i = 0;
		Imath::Euler<scalar_type> trialEuler( (rand()%(STEPS*2))*M_PI/(STEPS*2),
		                          (rand()%(STEPS*2))*M_PI/(STEPS*2),
		                          (rand()%(STEPS*2))*M_PI/(STEPS*2) );
		Imath::Matrix44<scalar_type> trialMat( trialEuler.toMatrix44() );
		scalar_type xScale = 0.1 * ( rand() % 100 );
		scalar_type yScale = 0.1 * ( rand() % 100 );
		scalar_type zScale = 0.1 * ( rand() % 100 );
		printf("dimensions = (%f, %f, %f)\n", xScale, yScale, zScale );
		for( std::vector<Imath::Vec3<scalar_type> >::iterator it = buffer.begin(); it != buffer.end(); ++it )
		{
			Imath::Vec3<scalar_type> basePoint( xScale*(i%2), yScale*((i/2)%2), zScale*(i/4));
			trialMat.multVecMatrix( basePoint, *it );
			++i;
		}
		OrientedBoundingBox<scalar_type> obb( buffer.begin(), buffer.end() );
		Imath::Vec3<scalar_type> n = obb.GetBox().min, x = obb.GetBox().max;
		printf("obb box is: (%f, %f, %f)-(%f, %f, %f)\n", n.x, n.y, n.z, x.x, x.y, x.z);
		Imath::Euler<scalar_type> e = obb.GetRotation();
		printf("obb rotation is: (%f, %f, %f)\n", e.x, e.y, e.z);
		printf("trial rotation is: (%f, %f, %f)\n", trialEuler.x, trialEuler.y, trialEuler.z);
		Imath::Matrix44<scalar_type> m = obb.GetTransform();
		printf("obb transform is:\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n",
			   m[0][0],m[0][1],m[0][2],m[0][3],
			   m[1][0],m[1][1],m[1][2],m[1][3],
			   m[2][0],m[2][1],m[2][2],m[2][3],
			   m[3][0],m[3][1],m[3][2],m[3][3] );
		printf("trial transform is:\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n(%f, %f, %f, %f)\n",
			   trialMat[0][0],trialMat[0][1],trialMat[0][2],trialMat[0][3],
			   trialMat[1][0],trialMat[1][1],trialMat[1][2],trialMat[1][3],
			   trialMat[2][0],trialMat[2][1],trialMat[2][2],trialMat[2][3],
			   trialMat[3][0],trialMat[3][1],trialMat[3][2],trialMat[3][3] );
	}
}
