/* Copyright  (C) 2010-2017 The RetroArch team
 *
 * ---------------------------------------------------------------------------------------
 * The following license statement only applies to this file (rthreads.c).
 * ---------------------------------------------------------------------------------------
 *
 * Permission is hereby granted, free of charge,
 * to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifdef __unix__
#define _POSIX_C_SOURCE 199309
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <boolean.h>
#include <rthreads/rthreads.h>

#include <switch.h>

struct sthread
{
   Thread id;
};

struct slock
{
   Mutex id;
};

struct scond
{
   CondVar id;
};

/**
 * sthread_create:
 * @start_routine           : thread entry callback function
 * @userdata                : pointer to userdata that will be made
 *                            available in thread entry callback function
 *
 * Create a new thread using the operating system's default thread
 * priority.
 *
 * Returns: pointer to new thread if successful, otherwise NULL.
 */
sthread_t *sthread_create(void (*thread_func)(void*), void *userdata)
{
	return sthread_create_with_priority(thread_func, userdata, 0);
}

static int next_core = 1;

/**
 * sthread_create_with_priority:
 * @start_routine           : thread entry callback function
 * @userdata                : pointer to userdata that will be made
 *                            available in thread entry callback function
 * @thread_priority         : thread priority hint value from [1-100]
 *
 * Create a new thread. It is possible for the caller to give a hint
 * for the thread's priority from [1-100]. Any passed in @thread_priority
 * values that are outside of this range will cause sthread_create() to
 * create a new thread using the operating system's default thread
 * priority.
 *
 * Returns: pointer to new thread if successful, otherwise NULL.
 */
sthread_t *sthread_create_with_priority(void (*thread_func)(void*), void *userdata, int thread_priority)
{
   sthread_t* thread = (sthread_t*)malloc(sizeof(sthread_t));
   if (!R_SUCCEEDED(threadCreate(&thread->id, thread_func, userdata, 1024*1024*4, 0x3B, next_core++ % 3)))
   {
      free(thread);
      return NULL;
   }
   threadStart(&thread->id);
   return thread;
}

/**
 * sthread_detach:
 * @thread                  : pointer to thread object
 *
 * Detach a thread. When a detached thread terminates, its
 * resources are automatically released back to the system
 * without the need for another thread to join with the
 * terminated thread.
 *
 * Returns: 0 on success, otherwise it returns a non-zero error number.
 */
int sthread_detach(sthread_t *thread)
{
   if (thread)
      return 1;
   printf("detached was called\n");
   return 1;
}

/**
 * sthread_join:
 * @thread                  : pointer to thread object
 *
 * Join with a terminated thread. Waits for the thread specified by
 * @thread to terminate. If that thread has already terminated, then
 * it will return immediately. The thread specified by @thread must
 * be joinable.
 *
 * Returns: 0 on success, otherwise it returns a non-zero error number.
 */
void sthread_join(sthread_t *thread)
{
   if (!thread)
      return;
   threadWaitForExit(&thread->id);
}

/**
 * sthread_isself:
 * @thread                  : pointer to thread object
 *
 * Returns: true (1) if calling thread is the specified thread
 */
bool sthread_isself(sthread_t *thread)
{
   /* This thread can't possibly be a null thread */
   if (!thread)
      return false;

   return threadGetCurHandle() == thread->id.handle;
}

/**
 * slock_new:
 *
 * Create and initialize a new mutex. Must be manually
 * freed.
 *
 * Returns: pointer to a new mutex if successful, otherwise NULL.
 **/
slock_t *slock_new(void)
{
   slock_t* res = (slock_t*)malloc(sizeof(slock_t));
   mutexInit(&res->id);
   return res;
}

/**
 * slock_free:
 * @lock                    : pointer to mutex object
 *
 * Frees a mutex.
 **/
void slock_free(slock_t *lock)
{
   if (!lock)
      return;
   free(lock);
}

/**
 * slock_lock:
 * @lock                    : pointer to mutex object
 *
 * Locks a mutex. If a mutex is already locked by
 * another thread, the calling thread shall block until
 * the mutex becomes available.
**/
void slock_lock(slock_t *lock)
{
   if (!lock)
      return;
   mutexLock(&lock->id);
}

/**
 * slock_unlock:
 * @lock                    : pointer to mutex object
 *
 * Unlocks a mutex.
 **/
void slock_unlock(slock_t *lock)
{
   if (!lock)
      return;
   mutexUnlock(&lock->id);
}

/**
 * scond_new:
 *
 * Creates and initializes a condition variable. Must
 * be manually freed.
 *
 * Returns: pointer to new condition variable on success,
 * otherwise NULL.
 **/
scond_t *scond_new(void)
{
   scond_t* res = (scond_t*)malloc(sizeof(scond_t));
   condvarInit(&res->id);
   return res;
}

/**
 * scond_free:
 * @cond                    : pointer to condition variable object
 *
 * Frees a condition variable.
**/
void scond_free(scond_t *cond)
{
   if (!cond)
      return;
   free(cond);
}

/**
 * scond_wait:
 * @cond                    : pointer to condition variable object
 * @lock                    : pointer to mutex object
 *
 * Block on a condition variable (i.e. wait on a condition).
 **/
void scond_wait(scond_t *cond, slock_t *lock)
{
   condvarWait(&cond->id, &lock->id);
}

/**
 * scond_broadcast:
 * @cond                    : pointer to condition variable object
 *
 * Broadcast a condition. Unblocks all threads currently blocked
 * on the specified condition variable @cond.
 **/
int scond_broadcast(scond_t *cond)
{
   return !R_SUCCEEDED(condvarWakeAll(&cond->id));
}

/**
 * scond_signal:
 * @cond                    : pointer to condition variable object
 *
 * Signal a condition. Unblocks at least one of the threads currently blocked
 * on the specified condition variable @cond.
 **/
void scond_signal(scond_t *cond)
{
   condvarWakeOne(&cond->id);
}

/**
 * scond_wait_timeout:
 * @cond                    : pointer to condition variable object
 * @lock                    : pointer to mutex object
 * @timeout_us              : timeout (in microseconds)
 *
 * Try to block on a condition variable (i.e. wait on a condition) until
 * @timeout_us elapses.
 *
 * Returns: false (0) if timeout elapses before condition variable is
 * signaled or broadcast, otherwise true (1).
 **/
bool scond_wait_timeout(scond_t *cond, slock_t *lock, int64_t timeout_us)
{
   return R_SUCCEEDED(condvarWaitTimeout(&cond->id, &lock->id, timeout_us * 1000));
}

#ifdef HAVE_THREAD_STORAGE
bool sthread_tls_create(sthread_tls_t *tls)
{
#ifdef USE_WIN32_THREADS
   return (*tls = TlsAlloc()) != TLS_OUT_OF_INDEXES;
#else
   return pthread_key_create((pthread_key_t*)tls, NULL) == 0;
#endif
}

bool sthread_tls_delete(sthread_tls_t *tls)
{
#ifdef USE_WIN32_THREADS
   return TlsFree(*tls) != 0;
#else
   return pthread_key_delete(*tls) == 0;
#endif
}

void *sthread_tls_get(sthread_tls_t *tls)
{
#ifdef USE_WIN32_THREADS
   return TlsGetValue(*tls);
#else
   return pthread_getspecific(*tls);
#endif
}

bool sthread_tls_set(sthread_tls_t *tls, const void *data)
{
#ifdef USE_WIN32_THREADS
   return TlsSetValue(*tls, (void*)data) != 0;
#else
   return pthread_setspecific(*tls, data) == 0;
#endif
}
#endif
