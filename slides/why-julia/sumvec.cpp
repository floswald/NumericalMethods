#include <iostream>
#include <vector>

int main(){

	// fill vector with 1,2,3,4,5
	std::vector<int> x;
	for (int i=1;i<5;i++){
		x.push_back(i);
	}
	// sum them up
	int sum = 0;
	for (std::vector<int>::iterator i=x.begin();i!=x.end();i++){
		sum += *i;
	}
	std::cout << "sum is " << sum << std::endl;
}


