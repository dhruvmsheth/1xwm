output_dir='data/genie_baseline_generated'
for i in $(seq 0 10 240); do
  python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M --output_dir "$output_dir" --example_ind "$i" --maskgit_steps 12 --temperature 0.8
  python visualize.py --token_dir "$output_dir"
  mv "$output_dir"/generated_offset0.gif "$output_dir"/example_"$i".gif
  mv "$output_dir"/generated_comic_offset0.png "$output_dir"/example_"$i".png
done
