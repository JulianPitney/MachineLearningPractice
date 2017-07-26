#include<iostream>



using namespace std;

void increment(int* counter) {
	cout << counter << endl;
	cout << *counter << endl;
	cout << endl;

	*counter = *counter + 1;
	cout << counter << endl;
	cout << *counter << endl;
	cout << endl;
}



int main() {

	int* counter = new int(10);
	cout << counter << endl;
	cout << *counter << endl;	
	cout << endl;
	increment(counter);
	cout << counter << endl;
	cout << *counter << endl;
	return 0;
}
