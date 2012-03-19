surface my_ptc(	varying float leafid = -1)
{
	normal Nn = normalize( faceforward( N, I ) );
	Ci = diffuse( Nn ) * cellnoise( point(leafid,0.0,0.0) );
	Oi = Os;
}
