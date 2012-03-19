/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

#ifndef rotate_point_around_line_h
#define rotate_point_around_line_h


//! adapted from http://local.wasp.uwa.edu.au/~pbourke/geometry/rotate/example.c
template< typename VEC, typename REAL >
__host__ __device__
VEC rotatePointAboutLine( VEC p,REAL theta,VEC p1,VEC p2)
{
   VEC u,q1,q2;
   REAL d;

   /* Step 1 */
   q1.x = p.x - p1.x;
   q1.y = p.y - p1.y;
   q1.z = p.z - p1.z;

   u.x = p2.x - p1.x;
   u.y = p2.y - p1.y;
   u.z = p2.z - p1.z;
   u = normalize(u);
   d = sqrtf(u.y*u.y + u.z*u.z);

   /* Step 2 */
   if (d != 0) {
      q2.x = q1.x;
      q2.y = q1.y * u.z / d - q1.z * u.y / d;
      q2.z = q1.y * u.y / d + q1.z * u.z / d;
   } else {
      q2 = q1;
   }

   /* Step 3 */
   q1.x = q2.x * d - q2.z * u.x;
   q1.y = q2.y;
   q1.z = q2.x * u.x + q2.z * d;

   /* Step 4 */
   q2.x = q1.x * cosf(theta) - q1.y * sinf(theta);
   q2.y = q1.x * sinf(theta) + q1.y * cosf(theta);
   q2.z = q1.z;

   /* Inverse of step 3 */
   q1.x =   q2.x * d + q2.z * u.x;
   q1.y =   q2.y;
   q1.z = - q2.x * u.x + q2.z * d;

   /* Inverse of step 2 */
   if (d != 0) {
      q2.x =   q1.x;
      q2.y =   q1.y * u.z / d + q1.z * u.y / d;
      q2.z = - q1.y * u.y / d + q1.z * u.z / d;
   } else {
      q2 = q1;
   }

   /* Inverse of step 1 */
   q1.x = q2.x + p1.x;
   q1.y = q2.y + p1.y;
   q1.z = q2.z + p1.z;
   return(q1);
}

#endif /* rotate_point_around_line_h */

