#include "plane.hpp"

using namespace dgal;


bool approx_same(const Imath::V2f a, const Imath::V2f b) { return (a-b).length() < 0.001f; }
bool approx_same(const Imath::V3f a, const Imath::V3f b) { return (a-b).length() < 0.001f; }


int main(int argc, char** argv)
{
	Imath::Plane3f plane(Imath::V3f(0,0,1), 0);

	std::vector<Imath::V3f> points(3);
	points[0] = Imath::V3f(4,5,8);
	points[1] = Imath::V3f(5,2,9);
	points[2] = Imath::V3f(2,-1,-3);
	Imath::V3f up(0,1,0);

	Projectionf_2D proj(plane, &up);

	std::vector<Imath::V2f> points2D;
	proj.project(points, points2D);

	for(unsigned int i=0; i<points2D.size(); ++i)
		std::cout << points2D[i] << std::endl;

	return 0;
}
