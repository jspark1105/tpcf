/**
Copyright (c) 2013, Intel Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    printf("Usage: ./generate out_file_name num_of_galaxies [random_seed]\n");
    printf("By default random_seed is 0\n");
  }

  FILE *file = fopen(argv[1], "w");
  if (NULL == file)
  {
    fprintf(stderr, "Failed to open file %s\n", argv[1]);
    return -1;
  }
  long long n = atoll(argv[2]);

  unsigned seed = argc >= 4 ? atoi(argv[3]) : 0;

  // header
  fputs("PCLLBL95123", file);
  fwrite(&n, sizeof(n), 1, file);

  // main
  srand(seed);
  for (long long i = 0; i < n*3; i++)
  {
    float f = (float)rand()/RAND_MAX;
    if (f < 0) f = 0;
    if (f > 1) f = 1;
    fwrite(&f, sizeof(f), 1, file);
  }

  fclose(file);

  return 0;
}
