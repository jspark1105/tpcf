/* Generates initial models for TPCF using real SDSS data
 * by Jose Fiestas (09.03.13)
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <string>

using namespace std;
#define NMAX 100000

float a[NMAX],d[NMAX],zz[NMAX],R[NMAX],c=2.99792458E5,H0=67.8; //H0 in (km/sec)/Mpc

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    printf("Usage: ./generate input_sdss_file_name out_file_namen number_of_galaxies\n");
/*    printf("By default random_seed is 0\n"); */
  }

  FILE *file_i = fopen(argv[1], "r");
  FILE *file_o = fopen(argv[2], "w");
  if (NULL == file_i)
  {
    fprintf(stderr, "Failed to open file %s\n", argv[1]);
    return -1;
  }
  long long n = atoll(argv[3]);

  unsigned seed = argc >= 5 ? atoi(argv[4]) : 0;

  // header
  fputs("PCLLBL95123", file_o);
  fwrite(&n, sizeof(n), 1, file_o);

 // read data
 
	ifstream file ( "sdss.csv" ); 
	int i=0;
	string zz_ch[NMAX],a_ch[NMAX],d_ch[NMAX];
	string::size_type sz;
	while ( file.good() )
	{
     	getline ( file, zz_ch[i],',');
     	getline ( file, a_ch[i],',');
     	getline ( file, d_ch[i]);
	istringstream(zz_ch[i]) >> zz[i];
	istringstream(a_ch[i]) >> a[i];
	istringstream(d_ch[i]) >> d[i];
cout<<i<<" "<<zz[i]<<" "<<a[i]<<" "<<d[i]<<endl;
	i++;
	}

/*
	ifstream inp (argv[2]); 
  for (int i = 0; i < n; i++){
	inp>>zz[i]>>a[i]>>d[i];
	} 
*/
 // write data
  for (long long i = 0; i < n; i++){
	a[i] = a[i] * M_PI/180;
	d[i] = d[i] * M_PI/180;
//	zz[i] = zz[i]*M_PI/180;
	R[i]=zz[i] * c/H0;
	float x=R[i]*cos(d[i])*cos(a[i]);
  x = (x + 26409)/52800;
  if (x < 0) x = 0;
  if (x > 1) x = 1;
	float y=R[i]*cos(d[i])*sin(a[i]);
  y = (y + 26409)/52800;
  if (y < 0) y = 0;
  if (y > 1) y = 1;
	float z=R[i]*sin(d[i]);
  z = (z + 26409)/52800;
  if (z < 0) z = 0;
  if (z > 1) z = 1;
	fwrite(&x,sizeof(x),1,file_o);
	fwrite(&y,sizeof(y),1,file_o);
	fwrite(&z,sizeof(z),1,file_o);
cout<<i<<" "<<x<<" "<<y<<" "<<z<<endl;
  }

  fclose(file_o);

  return 0;
}
