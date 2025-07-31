import React, { memo, useCallback } from 'react';
import { FixedSizeList as List } from 'react-window';
import { Exercise } from '../types';

interface ExerciseListProps {
  exercises: Exercise[];
  onExerciseSelect: (exercise: Exercise) => void;
}

interface ExerciseItemProps {
  index: number;
  style: React.CSSProperties;
  data: {
    exercises: Exercise[];
    onExerciseSelect: (exercise: Exercise) => void;
  };
}

// Memoized individual exercise item
const ExerciseItem = memo<ExerciseItemProps>(({ index, style, data }) => {
  const { exercises, onExerciseSelect } = data;
  const exercise = exercises[index];

  const handleClick = useCallback(() => {
    onExerciseSelect(exercise);
  }, [exercise, onExerciseSelect]);

  return (
    <div style={style} className="exercise-item">
      <div className="exercise-card" onClick={handleClick}>
        <h3>{exercise.name}</h3>
        <p>{exercise.description}</p>
        <div className="exercise-meta">
          <span className={`difficulty ${exercise.difficulty}`}>
            {exercise.difficulty}
          </span>
          <span className="category">{exercise.category}</span>
        </div>
        <div className="target-muscles">
          {exercise.targetMuscles.map(muscle => (
            <span key={muscle} className="muscle-tag">
              {muscle}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
});

ExerciseItem.displayName = 'ExerciseItem';

const ExerciseList: React.FC<ExerciseListProps> = memo(({ exercises, onExerciseSelect }) => {
  const itemData = {
    exercises,
    onExerciseSelect
  };

  const itemSize = 120; // Fixed height for each exercise item

  return (
    <div className="exercise-list">
      <h3>Available Exercises</h3>
      <List
        height={400}
        itemCount={exercises.length}
        itemSize={itemSize}
        itemData={itemData}
        width="100%"
        overscanCount={3} // Pre-render 3 items above and below viewport
      >
        {ExerciseItem}
      </List>
    </div>
  );
});

ExerciseList.displayName = 'ExerciseList';

export default ExerciseList;