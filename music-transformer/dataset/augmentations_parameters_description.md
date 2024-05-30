# Parameters description:

* **shift_volume**: parameters for pitching tones
    * **enabled**: bool whether enable this augmentation
    * **shift_range**: range of tones shifts to use
* **shift_tones**: parameters for shifting tones in smart way. Not implemented at this moment
    * **enabled**: bool whether enable this augmentation
* **move_octave**: parameters for moving octave
    * **enable**: bool whether enable this augmentation
    * **octaves_shift**: probable number of octaves to shift
    * **shifts_probabilities**: probabilities of each shift
* **change_tempo**: parameters for changing tempo
    * **enabled**: bool whether enable this augmentation
    * **mean**: mean of distribution
    * **scale**: scale of distribution
* **shift_velocity**: parameters for changing velocity
    * **enabled**: bool whether enable this augmentation
    * **shift_range**: range of coefficient to multiply velocity by