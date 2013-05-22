/* 
 * stack.h
 *
 * University of Edinburgh
 *
 * Modified by s0840449.
 * 
 * Stack containing an array of doubles.
 */

#ifndef STACK_H_
#define STACK_H_

typedef struct stack_node_tag stack_node;
typedef struct stack_tag stack;

struct stack_node_tag {
  double data[2];
  stack_node *next;
};

struct stack_tag {
  stack_node *top;
};

stack *new_stack();
void free_stack(stack *);

void push(double *, stack *);
double *pop (stack *);

int is_empty (stack *);

#endif  /* STACK_H_ */
