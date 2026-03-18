#!/usr/bin/env python3
"""
Nutritionist CLI - Recipe Rating and Nutrition Analysis Tool (src_alt version)

Usage:
    ./nutritionist.py milk, honey, jam
    ./nutritionist.py chicken lemon garlic

Provides:
    I. Rating forecast (bad/so-so/great)
    II. Nutrition facts with % daily values
    III. Top-3 similar recipes from Epicurious dataset

Adapted for data from src_alt/nutritionist.ipynb
"""

import sys
from pathlib import Path
from typing import List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from recipes import RatingPredictor, NutritionCalculator, RecipeSimilarityFinder


def parse_ingredients(args: List[str]) -> List[str]:
    """Parse command-line arguments into ingredient list."""
    if not args:
        return []

    # Join all arguments into single string
    input_str = ' '.join(args)

    # Check if comma-separated
    if ',' in input_str:
        ingredients = [ing.strip().lower() for ing in input_str.split(',')]
    else:
        # Space-separated
        ingredients = [ing.strip().lower() for ing in args]

    # Remove empty strings
    ingredients = [ing for ing in ingredients if ing]

    return ingredients


def print_usage():
    """Print usage instructions."""
    print("Usage: ./nutritionist.py <ingredients>")
    print("\nExamples:")
    print("  ./nutritionist.py milk, honey, jam")
    print("  ./nutritionist.py chicken lemon garlic")
    print("\nProvides:")
    print("  I.   Rating forecast (bad/so-so/great)")
    print("  II.  Nutrition facts with % daily values")
    print("  III. Top-3 similar recipes")


def format_output(rating_class: str, nutrition: dict, similar_recipes: List[dict]):
    """Format and print the complete output."""
    print("\n" + "=" * 70)
    print("NUTRITION ANALYSIS RESULTS")
    print("=" * 70)

    # I. Rating Forecast
    print("\nI. OUR FORECAST")
    print("-" * 70)
    print(f"You might find it tasty, but in our opinion, it is a {rating_class.upper()} idea")
    print(f"to have a dish with that list of ingredients.")
    print(f"(Predicted class: {rating_class})")

    # II. Nutrition Facts
    print("\n\nII. NUTRITION FACTS")
    print("-" * 70)

    if not nutrition:
        print("No nutrition data available for the provided ingredients.")
    else:
        for ingredient, data in nutrition.items():
            print(f"\n{ingredient.title()}")

            nutrients = data.get('nutrients', {})

            if not nutrients:
                print("  Nutrition data not available")
                continue

            # Calculate and display % daily values
            displayed_nutrients = []

            for nutrient, amount in nutrients.items():
                if amount > 0 and nutrient in NutritionCalculator.DAILY_VALUES:
                    daily_value = NutritionCalculator.DAILY_VALUES[nutrient]
                    pct_dv = (amount / daily_value) * 100

                    displayed_nutrients.append((nutrient, pct_dv))

            # Sort by percentage (highest first) and display
            displayed_nutrients.sort(key=lambda x: x[1], reverse=True)

            for nutrient_name, pct_dv in displayed_nutrients:
                print(f"  {nutrient_name} - {pct_dv:.0f}% of Daily Value")

    # III. Similar Recipes
    print("\n\nIII. TOP-3 SIMILAR RECIPES")
    print("-" * 70)

    if not similar_recipes:
        print("No similar recipes found. Try different ingredients!")
    else:
        for i, recipe in enumerate(similar_recipes, 1):
            title = recipe.get('title', 'Unknown')
            recipe_rating = recipe.get('rating', 0.0)
            url = recipe.get('url', 'N/A')
            similarity = recipe.get('similarity', 0.0)

            print(f"\n{i}. {title}")
            print(f"   Rating: {recipe_rating:.2f}/5.0")
            print(f"   Similarity: {similarity*100:.0f}%")
            print(f"   URL: {url}")

    print("\n" + "=" * 70)


def main():
    """Main entry point for the CLI application."""
    try:
        # Parse command-line arguments
        if len(sys.argv) < 2:
            print("Error: No ingredients provided.\n")
            print_usage()
            sys.exit(1)

        ingredients = parse_ingredients(sys.argv[1:])

        if not ingredients:
            print("Error: Could not parse ingredients.\n")
            print_usage()
            sys.exit(1)

        print(f"\nAnalyzing ingredients: {', '.join(ingredients)}")
        print("Please wait...")

        # Get paths relative to script location
        script_dir = Path(__file__).parent
        data_dir = script_dir / 'data'

        # 1. Initialize rating predictor
        predictor = RatingPredictor(
            model_path=str(data_dir / 'best_model.joblib')
        )

        # 2. Predict rating class
        rating_class = predictor.predict(ingredients)

        # 3. Calculate nutrition
        calculator = NutritionCalculator(
            nutrition_csv=str(data_dir / 'nutrition_dv.csv')
        )
        nutrition = calculator.calculate_recipe_nutrition(ingredients)

        # 4. Find similar recipes
        finder = RecipeSimilarityFinder(
            recipes_csv=str(data_dir / 'epi_r.csv'),
            urls_csv=str(data_dir / 'recipes_urls.csv')
        )
        finder.fit()
        similar_recipes = finder.find_similar(ingredients, top_n=3)

        # 5. Format and print results
        format_output(rating_class, nutrition, similar_recipes)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\nError: Required file not found.")
        print(f"Details: {str(e)}")
        print("\nMake sure you have run the research notebook (nutritionist.ipynb) first")
        print("to generate all required data files.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: An unexpected error occurred.")
        print(f"Details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
