/*
Author: Aniket Sangwan (180001005)
*/

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define double long double
#define pb push_back
#define F first
#define S second

int size; // No of threads

vector<double>fitness_scores; // Stores the fitness values of every individual
int w=1; // Current id of city

// Structure for City in TSP problem
struct City{

    int id; // Stores the id of the city (Unique for every city)
    int x,y; // Coordinates of the city

    // Constructor for cities
    City(int a,int b){
        id=w;
        x=a;
        y=b;
        w++;
    }

    // Calculates distance between two cities
    double distance(City c){

        double xdis=abs(x-c.x);
        double ydis=abs(y-c.y);
        double distance=sqrt(xdis*xdis+ydis*ydis);

        return distance;

    }

    // Prints the coordinates of the city
    void print_coord(){
        cout<<"("<<x<<", "<<y<<")\n";
    }

};

// Calculates the total distance of the route taken
double routeDistance(vector<City>route){

    double path_dist=0,fitness=0.0;

    // TSP starts and ends at the same place. So the initial city is inserted again
    route.pb(route[0]);

    // Calculates pairwise distance

    for(int i=0;i<route.size()-1;i++){

        City fromCity=route[i];
        City toCity=route[i+1];

        path_dist+=fromCity.distance(toCity);

    }

    return path_dist;

}

/* Returns fitness value of given route.
We aim to minimize the distance. So, Fitness is taken to be the inverse of distance,
because a larger fitness score is considered better 
*/
double routeFitness(vector<City>route){

    double fitness=1.0/routeDistance(route);
    return fitness;

}

// Randomly creates route using the city list. Randomly selects the order in which we visit the cities
vector<City> createRoute(vector<City>CityList){

    shuffle(CityList.begin(),CityList.end(),std::default_random_engine(rand()));

    return CityList;

}

// Looping through the create route function to generate full population
vector<vector<City>> initialPopulation(int pop_size, vector<City>CityList){

    vector< vector<City> >population;

    #pragma omp parallel
    {
        vector< vector<City> >population_private;

        #pragma omp for nowait
        for(int i=0;i<pop_size;i++)
            population_private.pb(createRoute(CityList));

        #pragma omp critical
        population.insert(population.end(),population_private.begin(),population_private.end());

    }
    return population;

    }

// Sorts the population according to their fitness score and returns in a new vector
vector<vector<City>> rankRoutes(vector<vector<City>>population){

    vector<pair<double,int>>fitnessResults; // Stores fitness value and inital route id
    fitnessResults.resize(population.size());

    #pragma omp parallel for 
    for(int i=0;i<population.size();i++){
        fitnessResults[i].F=routeFitness(population[i]);
        fitnessResults[i].S=i;
    }

    // Sorts in descending order wrt fitness value
    sort(fitnessResults.begin(),fitnessResults.end(),greater <pair<double, int>> ());
    
    vector<vector<City>>sorted_population; // Will store population sorted in decreasing fitness value
    sorted_population.resize(population.size());

    #pragma omp parallel for
    for(int i=0;i<fitnessResults.size();i++){

        sorted_population[i]=population[fitnessResults[i].S];
        fitness_scores[i]=(fitnessResults[i].F);

    }

    return sorted_population;

}

/*
 Using fitness proportionate selection to select the parents that will be used 
to create the next generation.

First,'elite_size' number of individuals with highest fitness are automatically carried to the next
generation, before applying any selection algorithm.

Then it completes the mating pool using fitness proportionate selection.

*/
vector<vector<City>> mating_Pool(vector<vector<City>>population, int elite_size){

    vector<vector<City>>matingPool; // Stores the mating pool
    matingPool.resize(fitness_scores.size());
   

    double tot_fitness=0; // Stores total fitness

    #pragma omp parallel
    { 
      // Using elitism
    #pragma omp for
    for(int i=0;i<elite_size;i++) 
        matingPool[i]=(population[i]); 

    #pragma omp for reduction(+:tot_fitness)
    for(int i=0;i<fitness_scores.size();i++)
        tot_fitness+=fitness_scores[i];
    }
    // Assigning fitness weighed probability
    for(int i=0;i<fitness_scores.size();i++){

        fitness_scores[i]=100*(fitness_scores[i]/tot_fitness);
        if(i!=0) fitness_scores[i]+=fitness_scores[i-1];

    }

    // Using fitness proportionate selection

    #pragma omp parallel for
    for(int i=0;i<fitness_scores.size()-elite_size;i++){

        double pick=(double)rand()/RAND_MAX;
        pick*=100;
        for(int j=0;j<fitness_scores.size();j++){

            if(j==fitness_scores.size()-1){
                matingPool[i+elite_size]=(population[j]);
                break;
            }
            if(j==fitness_scores.size()-1 || fitness_scores[j]<=pick && fitness_scores[j+1]>pick){
                matingPool[i+elite_size]=(population[j]);
                break;
            } 

        }
    }

    return matingPool;

}

// Using ordered crossover to breed for next generation. 
vector<City> breed(vector<City>parent1, vector<City>parent2){

    vector<City>child;

    // Randomly selecting a subset from first parent string
    int geneA=((double)rand()/RAND_MAX)*parent1.size();
    int geneB=((double)rand()/RAND_MAX)*parent1.size();

    int startGene=min(geneA, geneB);
    int endGene=max(geneA, geneB);

    int n=parent1.size();
    bool used[n+1]={0};

    // Filling the remainder of the route from second parent string in the order in which they appear.
    for(int i=startGene;i<endGene;i++){ 
        child.pb(parent1[i]);
        used[parent1[i].id]=1;
    }

    for(int i=0;i<parent2.size();i++){
        if(used[parent2[i].id]==0)
            child.pb(parent2[i]);
    }

    return child;

}

// Using mating pool to breed children. Using elitism to retain the best routes. 
vector<vector<City>> breedPopulation(vector<vector<City>>matingpool, int elite_size){

    vector<vector<City>>children;
    children.resize(matingpool.size());
    int length=matingpool.size()-elite_size;

    for(int i=0;i<elite_size;i++)
        children[i]=(matingpool[i]);

    shuffle(matingpool.begin(),matingpool.end(),std::default_random_engine(rand()));

    // Filling the rest of the generation
    #pragma omp parallel for
    for(int i=0;i<length;i++){

        vector<City>child=breed(matingpool[i],matingpool[matingpool.size()-i-1]);
        children[i+elite_size]=(child);

    }

    return children;

}

/* Using swap mutation to mutate.
Helps in avoiding local convergence
mutationRate-> Probability that two cities will swap their position
*/
vector<City> mutate(vector<City>&individual, double mutationRate){

    for(int swapped=0;swapped<individual.size();swapped++){

        double prob=(double)rand()/RAND_MAX;

        if(prob<mutationRate){

            double swapWith=(double)rand()/RAND_MAX;
            swapWith*=individual.size();

            swap(individual[swapped], individual[swapWith]);

        }

    }

    return individual;

}

// Using mutate function to mutate the complete population
vector<vector<City>> mutatePopulation( vector<vector<City>>population, double mutationRate){

    vector<vector<City>>mutatedPop;
    mutatedPop.resize(population.size());

    for(int ind=0;ind<population.size();ind++){
        vector<City>mutatedInd = mutate(population[ind],mutationRate);
        mutatedPop[ind]=(mutatedInd);
    }

    return mutatedPop;

}

// Produces a new generation using all the functions above
vector<vector<City>> nextGeneration( vector<vector<City>>currentGen, int elite_size, double mutationRate){

    // Rank the routes in the current generation
    vector<vector<City>> popRanked=rankRoutes(currentGen);

    // cout<<routeDistance(popRanked[0])<<", ";

    // Determining potential parents and creating mating pool
    vector<vector<City>> matingpool=mating_Pool(popRanked, elite_size);
 
    // Creating new generation
    vector<vector<City>> children=breedPopulation(matingpool, elite_size);
    
    // Applying mutation
    vector<vector<City>> nextGen=mutatePopulation(children, mutationRate);
   
    return nextGen;

}

void Genetic_Algo( vector<City>population, int popSize, int elite_size, double mutationRate, int generations){

    // Creating initial population from city list
    vector<vector<City>>pop=initialPopulation(popSize, population);

    cout<<"Initial Distance was: "<<routeDistance(pop[0])<<endl;

    for(int i=0;i<generations;i++){

        fitness_scores.clear();
        fitness_scores.resize(popSize);
        pop=nextGeneration(pop, elite_size, mutationRate);

    }


    cout<<"Final Distance is: "<<routeDistance(pop[0])<<endl;

}

int main(int argc, char **argv){
	ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);

    omp_set_num_threads(atoi(argv[1]));
	
    int no_of_cities=40; // No of cities
    int popSize=800;    // Population Size
    int eliteSize=19;   // Elite Size
    double mutationRate=0.01;   // Rate of mutation
    int generations=1000;   // No of generations

    vector<City>cityList; // Stores the initial list of cities

    // Assigning random coordinates to cities
    for(int i=0;i<no_of_cities;i++){
        double x=200*((double)rand()/RAND_MAX);
        double y=200*((double)rand()/RAND_MAX);
        cityList.pb(City(x,y));
    }

    double start_time=omp_get_wtime();

    Genetic_Algo(cityList, popSize, eliteSize, mutationRate, generations);

    double final_tot=omp_get_wtime()-start_time;
    // Prints time taken
    cout<<fixed<<setprecision(5) <<"Time Taken by openmp program: "<< final_tot <<" s "<<endl;
	
return 0;
}
