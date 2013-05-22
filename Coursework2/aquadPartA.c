/* PPLS Coursework 2 (Level 11), Part A. s0840449. */

/* Compilation/run instructions: 

Should compile/run with the commands given in the coursework:

/usr/lib64/openmpi/bin/mpicc -o aquadPartA aquadPartA.c stack.c stack.h
/usr/lib64/openmpi/bin/mpirun -c NUM_PROC ./aquadPartA
*/


/* Report:

The focus of my implementation is to reduce communication costs. This is
achieved by sending as few messages as possible, and by reducing the data sent
with each message.

The main body of my farmer is a loop that continues as long as there are either
tasks to complete or workers still processing. In each iteration, the farmer
first checks for incoming messages, then sends out any available tasks to idle
workers.

To check for incoming messages the farmer makes a single pass over the workers,
issuing a MPI_Iprobe to any worker who is currently noted as 'busy'. This is a
non-blocking call that checks if there are any messages from the worker. If the
MPI_Iprobe detects a message, a full MPI_Recv is issued to receive the message,
using the correct tag from the MPI_Status. Checking only busy workers avoids
deadlock, and using an MPI_Iprobe first avoids waiting on still-computing
workers. An alternative to the MPI_Iprobe would be to use MPI_Irecv, but this
would require tracking the MPI_Request object created by an immediate receive. I
also personally feel the probe/receive model is semantically cleaner.

To send out available tasks to workers the farmer again makes a pass over the
workers, skipping any that are busy. Once an idle worker for a task is found the
farmer uses an MPI_Isend to send two MPI_DOUBLEs to the worker, representing the
left and right coordinates. MPI_Send is not used as it would cause the farmer to
wait for the data to be copied to the MPI buffer. The buffer used for MPI_Isend
is a per-worker buffer that is not overridden until a message is received back
from the same worker (so is safe to use).

Once the main farmer loop ends it does a final pass through the workers, sending
each an empty message tagged with an EXIT_TAG. MPI_Send is used here as it is
acceptable now to block on each worker: if any of them don't receive the message
quite quickly there is something wrong! MPI_Isend could also be used as
MPI_Finalize is happy as long as all processes reach it 'eventually', although
that word is hardly well defined...

As a final (or perhaps more accurately, initial) optimization, if the farmer has
either no tasks to give out or no free workers to hand tasks to, a blocking
MPI_Probe is issued, causing the farmer to wait on a message from any source and
with any tag. This stops the farmer from busy waiting (assuming that MPI_Probe
uses a proper blocking mechanism) when it cannot do anything. This optimization
cannot deadlock: for the farmer loop to be entered at all there either must be
some task on the stack, or at least one worker still working.

Worker processes also have a main loop, that iterates until they receive a quit
message from the farmer. The loop begins with a blocking MPI_Probe to check for
a message from the farmer. A blocking call was used as a worker has nothing else
to do until the farmer has a message (task) for them. Once a message has
arrived, a MPI_Recv is issued to receive the message (using the correct tag from
the probe). If the message is tagged EXIT_TAG the worker immediately breaks out
of the loop and ends. Otherwise, if the message is a task the worker retrieves
the coordinates from the receive buffer and computes the quadrature
approximation. If the approximation results in an accepted area (i.e. the error
is <= EPSILON), the worker uses an MPI_Send to pass the calculated area back to
the farmer (as an MPI_DOUBLE). Again a blocking call is used as the worker has
nothing to do but wait. If however the result of the computation requires that
new tasks be scheduled, the worker sends an empty message back to the farmer,
with an appropriate tag. An empty message is used as the farmer does not
actually need any information from the worker. By caching the current
coordinates for each worker the farmer can calculate `mid' itself and thus
create the sub-tasks if the worker reports more accuracy is needed. This removes
the need to send three MPI_DOUBLEs in the message (or two messages!), thus
reducing communication costs.

*/

/* FUN FACT! All of aquadPartA.c is ANSI C89-compliant... I got bored. */

#include <mpi.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "stack.h"

/******************************************************************************
 * The following was originally in aquadPartA.h, but submit is set up to not  *
 * accept files other than aquadPartA.c, stack.c, and stack.h, so replicating *
 * it here. Messy...                                                          *
 ******************************************************************************/

#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0

#define SLEEPTIME 1

/* A structure for the farmer that holds information on a worker process. */
typedef struct worker_s {
  /* The worker's process id. */
  int id;

  /*
   * Storage for MPI_Requests tracking the status of the data sent to
   * the worker. As the code operates on strict turn-by-turn message passing
   * (i.e. a worker should not send a message until it has received one),
   * these are only used for error checking.
   */
  MPI_Request* request;

  /* A flag to note if the worker is busy computing a task or not. */
  bool busy;

  /*
   * The current coordinates that the worker is computing with. The contents
   * are only valid if busy is set.
   */
  double coordinates[2];
} worker;

/* Farmer methods. */
double farmer_main(int* tasks_per_process, int number_processes);
void recieve_data_from_workers(int number_of_workers, double* estimated_area,
    worker** workers, int* working_count, stack* task_stack);
void send_tasks_to_workers(int number_of_workers, worker** workers,
    int* working_count, stack* task_stack, int* tasks_per_process);

/* Worker methods. */
void worker_main(int worker_id);
void compute_quad(double left, double right, double* initial_estimated_area,
    double* left_area, double* right_area);

/*
 * Debug macros. To use them, define DEBUG when compiling, i.e. via -DDEBUG.
 *
 * Source:
 * http://stackoverflow.com/questions/1644868/c-define-macro-for-debug-printing
 */
void dbg_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}
#ifdef DEBUG
  #define DEBUG_MODE 1
#else
  #define DEBUG_MODE 0
#endif
#define TRACE(x) do { if (DEBUG_MODE) dbg_printf x; } while (0)


/*******************************************************************************
 * This marks the end of aquadPartA.h and the start of the actual aquadPartA.c *
 *******************************************************************************/

/* The farmer processor id. */
const int FARMER_ID = 0;

/* Tag definitions for MPI. */
const int EXIT_TAG = 0;
const int EXECUTE_TASK_TAG = 1;
const int RESULTS_NEW_TASKS_TAG = 2;
const int RESULTS_NEW_AREA_TAG = 3;

/*
 * The main method, responsible for setting up the adaptive quadrature
 * for both farmer and worker processes, and for outputting the results.
 */
int main(int argc, char **argv ) {
  int process_id;
  int number_processes;
  double approximated_area;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

  if(number_processes < 2) {
    fprintf(stderr, "ERROR: Must have at least 2 processes to run"
        " (found %d).\n", number_processes);
    MPI_Finalize();
    return 1;
  }

  if (process_id == FARMER_ID) {
    int i;

    /* The farmer must track how many tasks each worker processes. */
    int* tasks_per_process = (int*) malloc( sizeof(int) * number_processes);
    for (i = 0; i < number_processes; i++) {
      tasks_per_process[i] = 0;
    }

    /* Do the main calculation, farming out computations to workers. */
    approximated_area = farmer_main(tasks_per_process, number_processes);

    /* Dump the data. */
    printf("Area=%f\n", approximated_area);
    printf("\nTasks Per Process\n");
    for (i = 0; i < number_processes; i++) {
      printf("%d\t", i);
    }
    printf("\n");

    for (i = 0; i < number_processes; i++) {
      printf("%d\t", tasks_per_process[i]);
    }
    printf("\n");

    free(tasks_per_process);
  } else {
    worker_main(process_id);
  }

  MPI_Finalize();

  return 0;
}

/*
 * The main farmer method, which computes an adaptive quadrature using a
 * pool of workers.
 */
double farmer_main(int* tasks_per_process, int number_processes) {
  int i, working_count, number_of_workers;
  double approximate_area;
  double initial_points[2];
  stack* task_stack;
  worker** workers;

  number_of_workers = number_processes - 1;
  working_count = 0;

  /*
   * The necessary information for each worker is tracked, including their
   * id, whether they are busy or not, and the current task they are working
   * on.
   *
   * The MPI_Request for the most recent Isend that sent data to the worker is
   * used to sanity check the farmers actions.
   */
  workers = (worker**) malloc(sizeof(worker*) * number_of_workers);
  for (i = 0; i < number_of_workers; i++) {
    workers[i] = (worker*) malloc(sizeof(worker));

    /* The farmer is process 0, so worker ids are offset by 1. */
    workers[i]->id = i + 1;

    workers[i]->request = (MPI_Request*) malloc(sizeof(MPI_Request));
    workers[i]->busy = false;
    workers[i]->coordinates[0] = 0.0;
    workers[i]->coordinates[1] = 0.0;
  }

  /*
   * Store the tasks that are to be completed. Tasks are retrieved on a LIFO
   * basis, although the order is actually irrelevant for adaptive quadrature.
   */
  task_stack = new_stack();
  initial_points[0] = A;
  initial_points[1] = B;
  push(initial_points, task_stack);

  approximate_area = 0.0;
  while (!is_empty(task_stack) || working_count > 0) {
    if (working_count == number_of_workers || is_empty(task_stack)) {
      /*
       * There are either no tasks to hand out, or no free workers to receive
       * a task. In either case there is no point busy waiting for a worker
       * to free up.
       *
       * Note: This cannot deadlock unless a worker crashes. If there were no
       * workers left to send messages, working_count == 0 != number_workers.
       * Therefore, task_stack must be empty. But if that were true, the outer
       * loop test would fail, and the while loop would terminate before this
       * conditional.
       */
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    /*
     * Checking for incoming messages from workers before sending out new tasks
     * allows for a quick early-exit if there are no tasks to dispatch.
     */
    recieve_data_from_workers(number_of_workers, &approximate_area,
        workers, &working_count, task_stack);

    if (is_empty(task_stack)) {
      continue;
    }

    send_tasks_to_workers(number_of_workers, workers, &working_count,
        task_stack, tasks_per_process);
  }

  /* Terminate the workers. */
  for (i = 0; i < number_of_workers; i++) {
    MPI_Send(&i, 0, MPI_INT, workers[i]->id, EXIT_TAG, MPI_COMM_WORLD);
  }

  /* Free the array of worker information. */
  for (i = 0; i < number_of_workers; i++) {
    free(workers[i]->request);
    free(workers[i]);
  }
  free(workers);

  /* Free the stack. */
  free_stack(task_stack);

  return approximate_area;
}

/*
 * Receives data from any and all workers who have send data to the
 * farmer. Each worker is checked once, so if a worker has sent multiple
 * messages (which it actually can't in this program) this method would
 * only collect the first.
 *
 * This method should only be called by the farmer.
 */
void recieve_data_from_workers(int number_of_workers, double* estimated_area,
    worker** workers, int* working_count, stack* task_stack) {
  int i, message_waiting, worker_id, recieved_tag;
  MPI_Status status;

  for (i = 0; i < number_of_workers; i++) {
    worker_id = workers[i]->id;

    if (!workers[i]->busy) {
      /* The worker isn't doing anything, so won't have a message! */
      continue;
    }

    message_waiting = 0;
    MPI_Iprobe(worker_id, MPI_ANY_TAG, MPI_COMM_WORLD, &message_waiting,
        &status);
    recieved_tag = status.MPI_TAG;

    if (message_waiting) {
      /*
       * Sanity check: A worker should not be sending a message back if they
       * haven't received one yet.
       */
      int send_finished = false;
      MPI_Test(workers[i]->request, &send_finished, &status);
      if (!send_finished) {
        /*
         * There is no sense in letting messages pile up, so accept the 
         * message, but warn the user.
         */
        fprintf(stderr, "Farmer: ERROR! Message received before send!\n");
      }

      /*
       * Messages can either send back an area to add to the total, or new
       * tasks to be completed.
       */
      if (recieved_tag == RESULTS_NEW_AREA_TAG) {
        double new_area = 0.0;

        MPI_Recv(&new_area, 1, MPI_DOUBLE, worker_id, RESULTS_NEW_AREA_TAG,
            MPI_COMM_WORLD, &status);
        TRACE(("Farmer: Area received: %f.\n", new_area));

        (*estimated_area) += new_area;
      } else if (recieved_tag == RESULTS_NEW_TASKS_TAG) {
        double left, right, mid;
        double left_task[2], right_task[2];

        /* No message data is required to add a new task. */
        MPI_Recv(NULL, 0, MPI_INT, worker_id, RESULTS_NEW_TASKS_TAG,
            MPI_COMM_WORLD, &status);
        TRACE(("%s\n", "Farmer: New tasks received."));

        /* Calculate the new tasks to add to the stack. */
        left = workers[i]->coordinates[0];
        right = workers[i]->coordinates[1];
        mid = (left + right) / 2;

        left_task[0] = left;
        left_task[1] = mid;
        push(left_task, task_stack);

        right_task[0] = mid;
        right_task[1] = right;
        push(right_task, task_stack);
      } else {
        fprintf(stderr, "Farmer: Unknown tag received from worker %d: %d\n",
            worker_id, recieved_tag);
      }

      workers[i]->busy = false;
      (*working_count)--;
    }
  }
}

/*
 * Pops task data off the stack and sends it to the workers. A single
 * early-exit pass is made through the workers, so the number of tasks
 * dispatched per call is bounded both by the stack size and the number
 * of idle workers.
 *
 * This method should only be called by the farmer.
 */
void send_tasks_to_workers(int number_of_workers, worker** workers,
    int* working_count, stack* task_stack, int* tasks_per_process) {
  int i;

  for (i = 0; i < number_of_workers; i++) {
    int worker_id = workers[i]->id;
    double* data;

    if (workers[i]->busy) {
      continue;
    }

    data = pop(task_stack);
    if (data == NULL) {
      fprintf(stderr, "Farmer: ERROR! Data popped from stack was null!\n");
      continue;
    }
    workers[i]->coordinates[0] = data[0];
    workers[i]->coordinates[1] = data[1];
    free(data);

    /*
     * As workers[i]->coordinates cannot be changed until a message has been
     * received back from worker i, and workers should not send data until they
     * have received this message, it is safe to use workers[i]->coordinates as
     * a buffer for Isend.
     */
    MPI_Isend(workers[i]->coordinates, 2, MPI_DOUBLE, worker_id,
        EXECUTE_TASK_TAG, MPI_COMM_WORLD, workers[i]->request);

    /* Update the tracking information. */
    workers[i]->busy = true;
    (*working_count)++;
    tasks_per_process[worker_id]++;

    /* Early exit check. */
    if (is_empty(task_stack)) {
      break;
    }
  }
}

/*
 * The main worker method, which receives pairs of coordinates from the farmer
 * and calculates the adaptive quadrature between them. If the error for the
 * calculation is small enough the estimated area is returned to the farmer,
 * otherwise the farmer is notified that the coordinates should be split.
 */
void worker_main(int worker_id) {
  int recieved_tag;
  MPI_Status status;

  for (;;) {
    /* Wait for a command from the farmer. It is acceptable to block during
     * this probe, as the worker has nothing else to do. */
    MPI_Probe(FARMER_ID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    recieved_tag = status.MPI_TAG;

    if (recieved_tag == EXECUTE_TASK_TAG) {
      double left, right;
      double left_area, right_area, initial_estimated_area;
      double error;

      double data[2];
      MPI_Recv(data, 2, MPI_DOUBLE, FARMER_ID, EXECUTE_TASK_TAG,
          MPI_COMM_WORLD, &status);
      left = data[0];
      right = data[1];

      TRACE(("Worker %d received (%f, %f)\n", worker_id, data[0], data[1]));

      compute_quad(left, right, &initial_estimated_area, &left_area,
          &right_area);

      /* Sleep as specified by coursework. */
      usleep(SLEEPTIME);

      error = fabs((left_area + right_area) - initial_estimated_area);

      if (error > EPSILON) {
        /* Notify the farmer that the task should be split. */
        MPI_Send(NULL, 0, MPI_INT, FARMER_ID, RESULTS_NEW_TASKS_TAG,
            MPI_COMM_WORLD);
      } else {
        /* Send the computed area back to the farmer. */
        double area = left_area + right_area;
        MPI_Send(&area, 1, MPI_DOUBLE, FARMER_ID, RESULTS_NEW_AREA_TAG,
            MPI_COMM_WORLD);
      }
    } else if (recieved_tag == EXIT_TAG) {
      MPI_Recv(NULL, 0, MPI_INT, FARMER_ID, EXIT_TAG, MPI_COMM_WORLD, &status);
      break;
    } else {
      /* Warn the user about unknown tags. */
      MPI_Recv(NULL, 0, MPI_INT, FARMER_ID, recieved_tag, MPI_COMM_WORLD,
          &status);
      fprintf(stderr, "Worker %d: Unknown tag %d received.\n", worker_id,
          recieved_tag);
    }
  }
}

/*
 * Computes the quadrature estimate for a given set of coordinates
 * (left, right). The values of the estimated area between (left, right),
 * (left, mid), and (mid, right) are calculated and placed in the
 * relevant variables.
 *
 * Note that this method does not determine whether or not the error in the
 * split areas is low enough or not.
 */
void compute_quad(double left, double right, double* initial_estimated_area,
    double* left_area, double* right_area) {
  double fleft, fright, mid, fmid;

  fleft = F(left);
  fright = F(right);
  (*initial_estimated_area) = (fleft + fright) * (right - left) / 2;

  mid = (left + right) / 2;
  fmid = F(mid);

  (*left_area) = (fleft + fmid) * (mid - left) / 2;
  (*right_area) = (fmid + fright) * (right - mid) / 2;
}
