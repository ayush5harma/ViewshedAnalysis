#ifndef __CUST_KERNEL_H__
#define __CUST_KERNEL_H__

#define MAX_THREAD_COUNT 1024
#define M_PI       3.14159265358979323846

#define SMALLEST_GRADIENTD (- 9999999999999999999999.0)
/*this value is returned by findMaxValueWithinDist() is there is no
   key within that distance.  The largest double value is 1.7 E 308 */



#define RB_REDD (0)
#define RB_BLACKD (1)

#define NONWALKABLE 0
#define NOT_VISITED 1
#define OPEN 2
#define CLOSED 3

typedef int elev_t;
typedef unsigned char vs_t;

//typedef int vs_t;


#define ENTERING_EVENT 1
#define EXITING_EVENT -1
#define CENTER_EVENT 0

/// ----> x+
//	| y+
//	v
//__host__ __device__
struct event_t
{

	unsigned short int row, col;
	float angle;
	float dist;
	char type;
	//type of the event: ENTERING_EVENT,  EXITING_EVENT, CENTER_EVENT
};



enum AlgorithType
{
	SINGLE_THREAD_PER_AGENT_EXTERNAL_DEV_FUNCS = 0,
	MULTI_THREAD_PER_AGENT_EXTERNAL_DEV_FUNCS = 1
};

typedef struct Cost
{
	int F;
	int G;
	int index;
}Cost;

typedef struct
{
	int size;
	Cost* costs;
}PQ;


typedef struct
{
	float distance;
	float gradient;
} treeVal;



/*<===========================================>
   //public:
   //The value that's stored in the tree
   //Change this structure to avoid type casting at run time */
typedef struct
{
	/*this field is mandatory and cannot be removed.
	   //the tree is indexed by this "key". */
	float key;

	/*anything else below this line is optional */
	float gradient;
	float maxGradient;
} TreeValueD;


/*The node of a tree */
typedef struct TreeNodeD
{
	TreeValueD value;

	char color;

	struct TreeNodeD* left;
	struct TreeNodeD* right;
	struct TreeNodeD* parent;

} TreeNodeD;

typedef struct
{
	TreeNodeD* root;
	TreeNodeD* NIL;
} RBTreeD;

void resizeDeviceVector(int size);

void insertToEventValues(treeVal* values, int size, int currentSize);

void sortEventsWithThrust(event_t* events, long int eventSize);

void sortTreeWithThrust(int totalSize);

void findMaxInTreeUnsorted(int totalSize, float grad, float dist, treeVal& val);


void deletFromTreeWithThrust(float dist, int& totalSize);

void astarWrapper(short* statuses, int* parents, PQ* pqs, int width, int height, int agentSize, int* startIdxs, int* goalIdxs, AlgorithType type);


void calculateEventsWrapper(event_t* events, int minX, int maxX, int minY, int maxY, int observerX, int observerY);

void iterateOverEventsWrapper(event_t* events, elev_t* elev, vs_t* viewshed, int observerX, int observerY, int eventSize, int radiusX, elev_t observer_elev);

void testWrapper(vs_t* viewshed, int viewshedSize);

void iterateOverEventsWrapperPartialViewshed(event_t* events, elev_t* elev, vs_t* viewshed, int observerX, int observerY, int eventSize, int columnSize, elev_t observer_elev, int minX, int minY, int viewShedColumnSize);

void cudaR3Wrapper(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols);

void cudaR2Wrapper(vs_t* viewshed, elev_t* elev, elev_t observer_elev, int minX, int maxX, int minY, int maxY, int observerX, int observerY, int ncols);



#endif // #ifndef __CUST_KERNEL_H__