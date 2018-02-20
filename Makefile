container:
	rm -f singularity/keras-tf.img
	sudo singularity create --size 3000 singularity/keras-tf.img
	sudo singularity bootstrap singularity/keras-tf.img singularity/keras-tf.def
	sudo chown --reference=singularity/keras-tf.def singularity/keras-tf.img

lytic-rsync-dry-run:
	rsync -n -arvzP --delete --exclude-from=rsync.exclude -e "ssh -A -t hpc ssh -A -t lytic" ./ :project/viral-learning

lytic-rsync:
	rsync -arvzP --delete --exclude-from=rsync.exclude -e "ssh -A -t hpc ssh -A -t lytic" ./ :project/viral-learning
