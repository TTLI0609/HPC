SUBDIRS = V1_OpenMP V2_MPI
CLEANDIRS = $(SUBDIRS:%=clean-%)
subdirs:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

clean: $(CLEANDIRS)
$(CLEANDIRS): 
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: clean