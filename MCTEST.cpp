#include<iostream>
#include<pthread.h>
#include<cstdlib>
#include<mutex>
#include<condition_variable>
using namespace std;

#define NUM_THREADS 4

mutex mtx;
condition_variable cv;
string data;
bool ready = false;
bool processed = false;

void *do_stuff(void* threadID) {
	
	unique_lock<mutex> lk(mtx);
	cout << "Thread creates unique_lock using mtx...\n";
	cout << "Thread begins waiting for condition_variable..\n";
	cv.wait(lk, []{return ready;});

	cout << "Thread starts processing...\n";
	data += "after processing";
	processed = true;
	lk.unlock();
	cout << "Thread sets processed to true and unlocks mutex...\n";
	cv.notify_one();
	cout << "Thread notifies condition_variable...\n";
}


int main() {

	pthread_t threads[NUM_THREADS];
	int rc;
	int arg = 0;
	pthread_create(&threads[0], NULL, do_stuff, (void *)&arg);
	cout << "Thread is created...\n";

	data = "some data...";

	{
		lock_guard<mutex> lk(mtx);
		ready = true;
		cout << "Main creates lock_guard using mtx and sets ready=true...\n";
	}
	cv.notify_one();
	cout << "Main notifies condition_variable...\n";
	
	{
		unique_lock<mutex> lk(mtx);
		cout << "Main creates unique_lock using mtx and waits on condition_variable...\n";
		cv.wait(lk, []{return processed;});
		cout << "Main receives wake-up signal and terminates...\n";
		
	}

	return 0;
}
