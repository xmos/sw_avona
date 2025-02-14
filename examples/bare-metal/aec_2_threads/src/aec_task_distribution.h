#ifndef aec_task_distribution_h_
#define aec_task_distribution_h_

/**
 * @page This header defines the data structures used when distributing tasks across hardware threads.
 *
 * The task distribution scheme distributes tasks across hardware threads for 2 scenarios.
 *      1. Distribute multiple unique tasks across multiple HW threads. For example, for a 3 tasks, 2 threads configuration,
 *         distribute [task0, task1, task2] across [Thread0, Thread1].
 *      2. Distribute multiple (task, channel) pairs across multiple HW threads. For example, for a 3 tasks, 2 channels, 2 threads
 *      configuration, distribute [(task0, ch0), (task0, ch1), (task1, ch0), (task1, ch1), (task2, ch0), (task2,
 *      ch1)] across [Thread0, Thread1].
 *      Number of channels used when defining the (task, channel) pair is fixed to max(`AEC_MAX_Y_CHANNELS`,
 *      `AEC_MAX_X_CHANNELS`).
 */

/**
 * @brief Structure used when distributing tasks across hardware threads.
 */ 
typedef struct {
    /** Task index.*/
    int task;
    /** Flag indicating whether the task is active on that core. The task is run on the core only when is_active is set
     * to 1*/
    int is_active;
}par_tasks_t;

/**
 * @brief Structure used when distributing (task, channel) pairs across hardware threads.
 */
typedef struct {
    /** Task index.*/
    int task;
    /** Channel index.*/
    int channel;
    /** Flag indicating whether the (task, channel) pair is active on that core. The (task, channel) pair is run on the
     * core only when is_active is set to 1*/
    int is_active;
}par_tasks_and_channels_t;


#define AEC_THREAD_COUNT   2 /// <Number of hardware threads available

/** Number of iterations run on a given thread when distributing 2 tasks across AEC_THREAD_COUNT threads*/
#define AEC_2_TASKS_PASSES   (1)

/** Number of iterations run on a given thread when distributing 3 tasks across AEC_THREAD_COUNT threads*/
#define AEC_3_TASKS_PASSES   (2)

/** Number of iterations run on a given thread when distributing 3 tasks, max(AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS)
 * channels across AEC_THREAD_COUNT number of threads*/
#define AEC_3_TASKS_AND_CHANNELS_PASSES   (3)

/** Number of iterations run on a given thread when distributing 2 tasks, max(AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS)
 * channels across AEC_THREAD_COUNT number of threads*/
#define AEC_2_TASKS_AND_CHANNELS_PASSES   (2)

/** Number of iterations run on a given thread when distributing 1 task, max(AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS)
 * channels across AEC_THREAD_COUNT number of threads*/
#define AEC_1_TASKS_AND_CHANNELS_PASSES   (1)

typedef struct {
    /** task distribution definition for 3 tasks, max(AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS) channels, scheduled across
     * AEC_THREAD_COUNT threads */
    par_tasks_and_channels_t par_3_tasks_and_channels[AEC_THREAD_COUNT][AEC_3_TASKS_AND_CHANNELS_PASSES];
    /** task distribution definition for 2 tasks, max(AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS) channels, scheduled across
     * AEC_THREAD_COUNT threads */
    par_tasks_and_channels_t par_2_tasks_and_channels[AEC_THREAD_COUNT][AEC_2_TASKS_AND_CHANNELS_PASSES];
    /** task distribution definition for 1 task, max(AEC_MAX_Y_CHANNELS, AEC_MAX_X_CHANNELS) channels, scheduled across
     * AEC_THREAD_COUNT threads */
    par_tasks_and_channels_t par_1_tasks_and_channels[AEC_THREAD_COUNT][AEC_1_TASKS_AND_CHANNELS_PASSES];
    /** task distribution definition for 2 tasks, scheduled across AEC_THREAD_COUNT threads */
    par_tasks_t par_2_tasks[AEC_THREAD_COUNT][AEC_2_TASKS_PASSES];
    /** task distribution definition for 3 tasks, scheduled across AEC_THREAD_COUNT threads */
    par_tasks_t par_3_tasks[AEC_THREAD_COUNT][AEC_3_TASKS_PASSES];
}task_distribution_t;

#endif /* aec_task_distribution_h_ */
