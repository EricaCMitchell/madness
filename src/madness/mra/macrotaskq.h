/**
 \file macrotaskq.h
 \brief Declares the \c macrotaskq and MacroTaskBase classes
 \ingroup mra

 A MacroTaskq executes tasks on World objects, e.g. differentiation of a function or other
 arithmetic. Complex algorithms can be implemented.

 The universe world is split into subworlds, each of them executing macrotasks of the task queue.
 This improves locality and speedups for large number of compute nodes, by reducing communications
 within worlds.

 The user defines a macrotask (an example is found in test_vectormacrotask.cc), the tasks are
 lightweight and carry only bookkeeping information, actual input and output are stored in a
 cloud (see cloud.h)

 The user-defined macrotask is derived from MacroTaskIntermediate and must implement the run()
 method. A heterogeneous task queue is possible.

*/



#ifndef SRC_MADNESS_MRA_MACROTASKQ_H_
#define SRC_MADNESS_MRA_MACROTASKQ_H_

#include <madness/world/cloud.h>
#include <madness/world/world.h>

namespace madness {

/// base class
class MacroTaskBase {
public:

	typedef std::vector<std::shared_ptr<MacroTaskBase> > taskqT;

	MacroTaskBase() {}
	virtual ~MacroTaskBase() {};

	double priority=0.0;
	enum Status {Running, Waiting, Complete, Unknown} stat=Unknown;

	void set_complete() {stat=Complete;}
	void set_running() {stat=Running;}
	void set_waiting() {stat=Waiting;}

	bool is_complete() const {return stat==Complete;}
	bool is_running() const {return stat==Running;}
	bool is_waiting() const {return stat==Waiting;}

	virtual void run(World& world, Cloud& cloud, taskqT& taskq) = 0;
	virtual void cleanup() = 0;		// clear static data (presumably persistent input data)

    virtual void print_me(std::string s="") const {
        printf("this is task with priority %4.1f\n",priority);
    }
    double get_priority() const {return priority;}

};
//
//std::ostream& operator<<(std::ostream& os, const MacroTaskBase::Status s) {
//	if (s==MacroTaskBase::Status::Running) os << "Running";
//	if (s==MacroTaskBase::Status::Waiting) os << "Waiting";
//	if (s==MacroTaskBase::Status::Complete) os << "Complete";
//	if (s==MacroTaskBase::Status::Unknown) os << "Unknown";
//	return os;
//}

template<typename macrotaskT>
class MacroTaskIntermediate : public MacroTaskBase {

public:

	MacroTaskIntermediate() {}

	~MacroTaskIntermediate() {}

	void run(World& world, Cloud& cloud) {
		dynamic_cast<macrotaskT*>(this)->run(world,cloud);
		world.gop.fence();
	}

	void cleanup() {};
};


class MacroTaskQ : public WorldObject< MacroTaskQ> {

    World& universe;
    std::shared_ptr<World> subworld_ptr;
	MacroTaskBase::taskqT taskq;
	std::mutex taskq_mutex;
	long printlevel=0;

	bool printdebug() const {return printlevel>=10;}
    bool printtimings() const {return universe.rank()==0 and printlevel>=3;}

public:

	madness::Cloud cloud;
	World& get_subworld() {return *subworld_ptr;}
	void set_printlevel(const long p) {printlevel=p;}

    /// create an empty taskq and initialize the subworlds
	MacroTaskQ(World& universe, int nworld, const long printlevel=0)
		  : universe(universe), WorldObject<MacroTaskQ>(universe), taskq(), cloud(universe), printlevel(printlevel) {

		subworld_ptr=create_worlds(universe,nworld);
		this->process_pending();
	}

	~MacroTaskQ() {}

	/// for each process create a world using a communicator shared with other processes by round-robin
	/// copy-paste from test_world.cc
	static std::shared_ptr<World> create_worlds(World& universe, const std::size_t nsubworld) {

		int color = universe.rank() % nsubworld;
		SafeMPI::Intracomm comm = universe.mpi.comm().Split(color, universe.rank() / nsubworld);

		std::shared_ptr<World> all_worlds;
		all_worlds.reset(new World(comm));

		universe.gop.fence();
		return all_worlds;
	}

	/// run all tasks, tasks may store the results in the cloud
	void run_all(MacroTaskBase::taskqT vtask=MacroTaskBase::taskqT()) {

		for (const auto& t : vtask) if (universe.rank()==0) t->set_waiting();
		for (int i=0; i<vtask.size(); ++i) add_replicated_task(vtask[i]);
		if (printdebug()) print_taskq();

        universe.gop.fence();
        universe.gop.set_forbid_fence(true); // make sure there are no hidden universe fences
        set_pmap(get_subworld());

        double cpu00=cpu_time();

		World& subworld=get_subworld();
		if (printdebug()) print("I am subworld",subworld.id());
		double tasktime=0.0;
		while (true){
			long element=get_scheduled_task_number(subworld);
            double cpu0=cpu_time();
			if (element<0) break;
			std::shared_ptr<MacroTaskBase> task=taskq[element];
            if (printdebug()) print("starting task no",element, "in subworld",subworld.id(),"at time",wall_time());

			task->run(subworld,cloud, taskq);

			double cpu1=cpu_time();
            set_complete(element);
			tasktime+=(cpu1-cpu0);
			if (subworld.rank()==0 and printlevel>=3) printf("completed task %3ld after %6.1fs at time %6.1fs\n",element,cpu1-cpu0,wall_time());

		}
        universe.gop.set_forbid_fence(false);
		universe.gop.fence();
		universe.gop.sum(tasktime);
        double cpu11=cpu_time();
        if (printtimings()) {
            printf("completed taskqueue after    %4.1fs at time %4.1fs\n", cpu11 - cpu00, wall_time());
            printf(" total cpu time / per world  %4.1fs %4.1fs\n", tasktime, tasktime / universe.size());
        }

		// cleanup task-persistent input data
		for (auto& task : taskq) task->cleanup();
		cloud.clear_cache(subworld);
		subworld.gop.fence();
        subworld.gop.fence();
        universe.gop.fence();
        universe.gop.fence();
        set_pmap(universe);
	}

	void add_tasks(MacroTaskBase::taskqT& vtask) {
        for (const auto& t : vtask) {
            if (universe.rank()==0) t->set_waiting();
            add_replicated_task(t);
        }
	}

//	/// run the task on the vector of input data, return vector of results
//	template<typename taskT>
//	std::vector<typename taskT::result_type> map(taskT& task1,
//			std::vector<typename taskT::data_type>& vdata) {
//
//		// create copies of the input task instance and fill with the data
//		std::vector<std::shared_ptr<MacroTaskBase> > vtask(vdata.size());
//		for (int i=0; i<vdata.size(); ++i) {
//			vtask[i]=std::shared_ptr<MacroTaskBase>(new taskT(task1));
//			dynamic_cast<taskT&>(*vtask[i].get()).set_data(vdata[i]);
//		}
//
//		// execute the task list
//		run_all(vtask);
//
//		// localize the result into universe
//		std::vector<typename taskT::result_type> vresult(vdata.size());
//		for (int i=0; i<vresult.size(); ++i) {
//			vtask[i]->load_result(universe,"result_of_task"+std::to_string(i));
//			vresult[i]=dynamic_cast<taskT&>(*(vtask[i].get())).get_result();
//		}
//		return vresult;
//	}

private:
	void add_replicated_task(const std::shared_ptr<MacroTaskBase>& task) {
		taskq.push_back(task);
	}

	void print_taskq() const {
		universe.gop.fence();
		if (universe.rank()==0) {
			print("\ntaskq on universe rank",universe.rank());
			for (const auto& t : taskq) t->print_me();
		}
		universe.gop.fence();
	}

	/// scheduler is located on universe.rank==0
	long get_scheduled_task_number(World& subworld) {
		long number=0;
		if (subworld.rank()==0) number=this->send(ProcessID(0), &MacroTaskQ::get_scheduled_task_number_local);
		subworld.gop.broadcast_serializable(number, 0);
		subworld.gop.fence();
		return number;

	}

	long get_scheduled_task_number_local() {
		MADNESS_ASSERT(universe.rank()==0);
		std::lock_guard<std::mutex> lock(taskq_mutex);

		auto is_Waiting = [](const std::shared_ptr<MacroTaskBase>& mtb_ptr) {return mtb_ptr->is_waiting();};
		auto it=std::find_if(taskq.begin(),taskq.end(),is_Waiting);
		if (it!=taskq.end()) {
			it->get()->set_running();
			long element=it-taskq.begin();
			return element;
		}
//		print("could not find task to schedule");
		return -1;
	}

	/// scheduler is located on rank==0
	void set_complete(const long task_number) const {
		this->task(ProcessID(0), &MacroTaskQ::set_complete_local, task_number);
	}

	/// scheduler is located on rank==0
	void set_complete_local(const long task_number) const {
		MADNESS_ASSERT(universe.rank()==0);
		taskq[task_number]->set_complete();
	}

public:
	void static set_pmap(World& world) {
        FunctionDefaults<1>::set_default_pmap(world);
        FunctionDefaults<2>::set_default_pmap(world);
        FunctionDefaults<3>::set_default_pmap(world);
        FunctionDefaults<4>::set_default_pmap(world);
        FunctionDefaults<5>::set_default_pmap(world);
        FunctionDefaults<6>::set_default_pmap(world);
	}
private:

	std::size_t size() const {
		return taskq.size();
	}

//	void add_task(const std::shared_ptr<MacroTaskBase>& task) {
//		ProcessID master=0;
//		task->print_me("in add_task, universe.rank()="+std::to_string(universe.rank()));
//		MacroTaskBase* taskptr=task.get();
//		thistype::send(master,&thistype::add_task_local,taskptr);
//	};
//
//	void add_task_local(const basetaskptr& task) {
//		MADNESS_ASSERT(universe.rank()==0);
//		std::shared_ptr<MacroTaskBase> task1;
//		task1.reset(task);
//		task1->print_me("in add_task_local");
//		taskq.push(task1);
//	};
//
//	std::shared_ptr<MacroTaskBase> get_task_from_tasklist(World& regional) {
//
//		// only root may pop from the task list
//		std::vector<unsigned char> buffer;
//		if (regional.rank()==0) buffer=pop();
//		regional.gop.broadcast_serializable(buffer, 0);
//		regional.gop.fence();
//
//		std::shared_ptr<MacroTaskBase> task;
//		MacroTaskBase* task_ptr;
//		BufferInputArchive ar(&buffer[0],buffer.size());
//		ar & task_ptr;
//
//		task.reset(task_ptr);
//		return task;
//	}
//
//	/// pass serialized task from universe.rank()==0 to world.rank()==0
//	std::vector<unsigned char> pop() {
//		return this->task(ProcessID(0), &macro_taskq<taskT>::pop_local);
//	}
//
//	/// pop highest-priority task and return it as serialized buffer
//	std::vector<unsigned char> pop_local() {
//		const std::lock_guard<std::mutex> lock(taskq_mutex);
//		std::shared_ptr<MacroTaskBase> task(NULL);
//
//		if (not taskq.empty()) {
//			task=taskq.top();
//			taskq.pop();
//		}
//
//		BufferOutputArchive ar_c;
//		ar_c & task.get();
//		long nbyte=ar_c.size();
//		std::vector<unsigned char> buffer(nbyte);
//
//		BufferOutputArchive ar2(&buffer[0],buffer.size());
//		ar2 & task.get();
//
//		return buffer;
//	}

};




} /* namespace madness */

#endif /* SRC_MADNESS_MRA_MACROTASKQ_H_ */
