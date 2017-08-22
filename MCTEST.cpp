#include<iostream>
#include<pthread.h>
#include<cstdlib>


using namespace std;

#define NUM_THREADS 4


void *do_stuff(void* threadID) {

	int i = 0;

	for(;;)
	{
		i = i * i + i * i;	
	}

	pthread_exit(NULL);
}


int main() {

	pthread_t threads[NUM_THREADS];
	int rc;

	for(int i = 0; i < NUM_THREADS; i++)
	{
		pthread_create(&threads[i], NULL, do_stuff, (void *)i);
	}



	pthread_exit(NULL);
}
